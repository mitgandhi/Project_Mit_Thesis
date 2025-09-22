import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for parallel plotting
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.stats import pearsonr
import os
import time

# Import helper functions from the local T1 module while remaining compatible
# with direct script execution.
try:
    from .T1 import (
        load_matlab_txt,
        round2,
        parse_geometry,
        parse_operating_conditions,
        create_cumulative_pressure_map,
        collect_simulation_data,
        create_comparison_plots,
        create_overall_summary,
        run_contact_pressure_analysis,
    )
except (ImportError, ValueError):
    from T1 import (  # type: ignore
        load_matlab_txt,
        round2,
        parse_geometry,
        parse_operating_conditions,
        create_cumulative_pressure_map,
        collect_simulation_data,
        create_comparison_plots,
        create_overall_summary,
        run_contact_pressure_analysis,
    )

# --- Regex Patterns for Folder Name Parsing ---
PARAM_PATTERN_NEW = re.compile(
    r"CL(?P<CL>[-\d\.]+)_dZ(?P<dZ>[-\d\.]+)_LKG(?P<LKG>[-\d\.]+)_lF(?P<lF>[-\d\.]+)_zeta(?P<zeta>[-\d\.]+)"
)

PARAM_PATTERN_OLD = re.compile(
    r"dK(?P<dK>[-\d\.]+)_dZ(?P<dZ>[-\d\.]+)_LKG(?P<LKG>[-\d\.]+)_lF(?P<lF>[-\d\.]+)_zeta(?P<zeta>[-\d\.]+)"
)


def detect_folder_pattern(folder_name: str) -> Tuple[str, Optional[re.Match]]:
    """
    Detect which pattern the folder name matches and return pattern type and match object.

    Returns:
        Tuple of (pattern_type, match_object) where pattern_type is "new", "old", or "none"
    """
    # Try new pattern first (with CL)
    match_new = PARAM_PATTERN_NEW.search(folder_name)
    if match_new:
        return "new", match_new

    # Try old pattern (with dK)
    match_old = PARAM_PATTERN_OLD.search(folder_name)
    if match_old:
        return "old", match_old

    return "none", None


def parse_geometry_file(geometry_file_path: str) -> Dict[str, Optional[float]]:
    """
    Extract dK and lF values from geometry.txt file.
    """
    result = {"dK": None, "lF": None}

    try:
        if not os.path.exists(geometry_file_path):
            return result

        with open(geometry_file_path, 'r') as file:
            content = file.read()

        # Look for dK parameter in the geometry file
        dk_pattern = re.compile(r'dK\s+([+-]?(?:\d+\.?\d*|\.\d+))')
        dk_match = dk_pattern.search(content)

        # Look for lF parameter in the geometry file
        lf_pattern = re.compile(r'lF\s+([+-]?(?:\d+\.?\d*|\.\d+))')
        lf_match = lf_pattern.search(content)

        if dk_match:
            result["dK"] = float(dk_match.group(1))

        if lf_match:
            result["lF"] = float(lf_match.group(1))

        return result

    except Exception as e:
        print(f"Error reading geometry file {geometry_file_path}: {e}")
        return result


def parse_parameters_flexible(folder_name: str, folder_path: str) -> Dict[str, float]:
    """
    Flexible parameter parsing that handles both old and new folder naming conventions.

    For new format (with CL): Extract dK from geometry.txt, calculate clearance
    For old format (with dK): Calculate clearance as dZ - dK
    """
    pattern_type, match = detect_folder_pattern(folder_name)

    if pattern_type == "none":
        return {}

    try:
        params = {k: float(v) for k, v in match.groupdict().items()}
        params["zeta"] = float(int(params["zeta"]))  # Keep zeta as integer

        # Extract geometry parameters (always try to get dK and lF from file)
        geometry_file_path = os.path.join(folder_path, 'input', 'geometry.txt')
        geometry_params = parse_geometry_file(geometry_file_path)

        if pattern_type == "new":
            # NEW FORMAT: Folder has CL, need to extract dK from geometry file
            if geometry_params["dK"] is not None:
                params["dK"] = geometry_params["dK"]

                # Calculate clearance from extracted dK and parsed dZ
                calculated_clearance = params["dZ"] - params["dK"]
                parsed_clearance = params["CL"]

                # Verify clearance values match (with tolerance)
                if abs(calculated_clearance - parsed_clearance) > 0.001:  # 1 micron tolerance
                    print(f"Clearance mismatch in {folder_name}")
                    print(f"     Parsed CL: {parsed_clearance:.6f} mm")
                    print(f"     Calculated (dZ-dK): {calculated_clearance:.6f} mm")

                # Use the calculated clearance based on extracted dK
                params["clearance"] = calculated_clearance

            else:
                print(f"Could not extract dK for {folder_name}, using parsed clearance")
                params["dK"] = params["dZ"] - params["CL"]  # Fallback calculation
                params["clearance"] = params["CL"]

        elif pattern_type == "old":
            # OLD FORMAT: Folder has dK, calculate clearance as dZ - dK
            # dK is already in params from folder name
            params["clearance"] = params["dZ"] - params["dK"]

            # If geometry file has dK, compare with folder value
            if geometry_params["dK"] is not None:
                geometry_dK = geometry_params["dK"]
                folder_dK = params["dK"]

                if abs(geometry_dK - folder_dK) > 0.001:
                    print(f"dK mismatch in {folder_name}")
                    print(f"     Folder dK: {folder_dK:.6f} mm")
                    print(f"     Geometry dK: {geometry_dK:.6f} mm")
                    print(f"     Using geometry file value")

                    # Use geometry file value and recalculate clearance
                    params["dK"] = geometry_dK
                    params["clearance"] = params["dZ"] - params["dK"]

        # Handle lF parameter (always prefer geometry file if available)
        if geometry_params["lF"] is not None:
            geometry_lF = geometry_params["lF"]
            folder_lF = params.get("lF", None)

            if folder_lF is not None and abs(geometry_lF - folder_lF) > 0.001:
                print(f"Using lF from geometry file for {folder_name}: {geometry_lF:.3f} mm")

            params["lF"] = geometry_lF

        return params

    except ValueError as e:
        print(f"Error parsing parameters from {folder_name}: {e}")
        return {}


def load_contact_pressure(sim_path: str) -> Dict[str, float]:
    """Load contact pressure data from Piston_Contact_Pressure.txt file"""
    contact_pressure_path = Path(sim_path) / "output" / "piston" / "matlab" / "Piston_Contact_Pressure.txt"

    if not contact_pressure_path.exists():
        return {"max_contact_pressure": 0.0, "avg_contact_pressure": 0.0, "contact_pressure_valid": False}

    try:
        # Load contact pressure data using the T1 function
        pressure_data = load_matlab_txt(str(contact_pressure_path))

        if pressure_data is None or pressure_data.size == 0:
            return {"max_contact_pressure": 0.0, "avg_contact_pressure": 0.0, "contact_pressure_valid": False}

        # Calculate statistics
        max_pressure = np.max(pressure_data)
        avg_pressure = np.mean(pressure_data)

        # Convert from Pa to MPa for easier interpretation
        max_pressure_mpa = max_pressure / 1e6
        avg_pressure_mpa = avg_pressure / 1e6

        return {
            "max_contact_pressure": max_pressure_mpa,
            "avg_contact_pressure": avg_pressure_mpa,
            "contact_pressure_valid": True
        }

    except Exception as e:
        print(f"Error loading contact pressure data from {sim_path}: {e}")
        return {"max_contact_pressure": 0.0, "avg_contact_pressure": 0.0, "contact_pressure_valid": False}


def calculate_contact_pressure_penalty(max_contact_pressure_mpa: float,
                                       pressure_threshold_mpa: float = 100.0,
                                       max_penalty_percent: float = 200.0) -> float:
    """
    Calculate contact pressure penalty based on maximum contact pressure.

    Args:
        max_contact_pressure_mpa: Maximum contact pressure in MPa
        pressure_threshold_mpa: Pressure threshold (default 100 MPa)
        max_penalty_percent: Maximum penalty percentage (200% by default)

    Returns:
        Penalty as a decimal (e.g., 2.0 for 200% penalty)
    """
    if max_contact_pressure_mpa <= 0:
        return 0.0

    # Linear scaling: 0% penalty at 0 pressure, max_penalty_percent at pressure_threshold
    penalty_percent = min(max_penalty_percent,
                          (max_contact_pressure_mpa / pressure_threshold_mpa) * max_penalty_percent)

    return penalty_percent / 100.0  # Convert percentage to decimal


def parse_loss_file(sim_path: str) -> Dict[str, float]:
    """Parse loss data from piston.txt file"""
    piston_path = Path(sim_path) / "output" / "piston" / "piston.txt"
    if not piston_path.exists():
        return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
    try:
        df_loss = pd.read_csv(piston_path, delimiter="\t")
        df_loss = df_loss[df_loss["revolution"] <= 6.0]
        if df_loss.empty:
            return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
        mech = abs(df_loss["Total_Mechanical_Power_Loss"].max())
        vol = abs(df_loss["Total_Volumetric_Power_Loss"].max())
        if pd.isna(mech) or pd.isna(vol):
            return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
        return {"mechanical": mech, "volumetric": vol, "total": mech + vol, "valid": True}
    except Exception as e:
        print(f"Error parsing loss file {sim_path}: {e}")
        return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}


def find_pareto_front(df: pd.DataFrame, objective_cols: list, minimize: bool = True) -> pd.DataFrame:
    """
    Identifies the Pareto front from a DataFrame of solutions.
    """
    if df.empty or not all(col in df.columns for col in objective_cols):
        return pd.DataFrame()

    df_sorted = df.sort_values(by=objective_cols[0], ascending=minimize).reset_index(drop=True)

    pareto_indices = []
    for i in range(len(df_sorted)):
        is_pareto = True
        for j in range(len(df_sorted)):
            if i == j:
                continue

            dominates_all_objectives = True
            strictly_better_in_one = False

            for obj in objective_cols:
                if minimize:
                    if df_sorted.loc[j, obj] > df_sorted.loc[i, obj]:
                        dominates_all_objectives = False
                        break
                    if df_sorted.loc[j, obj] < df_sorted.loc[i, obj]:
                        strictly_better_in_one = True
                else:
                    if df_sorted.loc[j, obj] < df_sorted.loc[i, obj]:
                        dominates_all_objectives = False
                        break
                    if df_sorted.loc[j, obj] > df_sorted.loc[i, obj]:
                        strictly_better_in_one = True

            if dominates_all_objectives and strictly_better_in_one:
                is_pareto = False
                break

        if is_pareto:
            pareto_indices.append(i)

    return df_sorted.loc[pareto_indices].drop_duplicates().reset_index(drop=True)


# --- Parallel Data Loading Functions ---

def process_single_simulation(sim_folder_data):
    """Process a single simulation folder - designed for parallel execution."""
    sim_path, iter_num, iter_type, opt_type, pressure_threshold_mpa, max_penalty_percent = sim_folder_data

    pattern_type, _ = detect_folder_pattern(sim_path.name)
    if pattern_type == "none":
        return None

    params = parse_parameters_flexible(sim_path.name, str(sim_path))
    if not params:
        return None

    # Load losses
    losses = parse_loss_file(str(sim_path))

    # Load contact pressure data
    contact_pressure_data = load_contact_pressure(str(sim_path))

    # Calculate penalty
    penalty = calculate_contact_pressure_penalty(
        contact_pressure_data["max_contact_pressure"],
        pressure_threshold_mpa,
        max_penalty_percent
    )

    # Calculate penalized total loss
    penalized_total = losses["total"] * (1 + penalty)

    # Combine all data
    record = {**params, **losses, **contact_pressure_data}
    record["contact_pressure_penalty"] = penalty
    record["penalized_total"] = penalized_total
    record[iter_type] = iter_num
    record["folder_name"] = str(sim_path)
    record["optimizer"] = opt_type
    record["pattern_type"] = pattern_type

    if record.get("valid") and record["total"] < 1e6:
        return record
    return None


def load_results_with_contact_pressure_parallel(folder_path: str, opt_type: str,
                                                pressure_threshold_mpa: float = 100.0,
                                                max_penalty_percent: float = 200.0,
                                                max_workers: int = None) -> list:
    """Parallel version of load_results function with contact pressure penalty calculation."""
    base_folder = Path(folder_path)
    if not base_folder.exists():
        print(f"Base folder not found: {base_folder}")
        return []

    subfolders = [f for f in base_folder.iterdir() if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders in {base_folder}")

    # Collect all simulation tasks
    simulation_tasks = []

    for folder in subfolders:
        folder_name = folder.name
        if opt_type == "NSGA-III":
            if folder_name.startswith("Generation_G"):
                try:
                    iter_num = int(folder_name.replace("Generation_G", ""))
                    iter_type = "generation"
                except ValueError:
                    continue
            elif folder_name == "Initial_Sampling":
                iter_num = 0
                iter_type = "generation"
            else:
                continue
        elif opt_type == "BO":
            if folder_name.startswith("Iteration_I"):
                try:
                    iter_num = int(folder_name.replace("Iteration_I", ""))
                    iter_type = "iteration"
                except ValueError:
                    continue
            elif folder_name == "Initial_Sampling":
                iter_num = 0
                iter_type = "iteration"
            else:
                continue
        else:
            continue

        sim_folders = [f for f in folder.iterdir() if f.is_dir()]
        for sim in sim_folders:
            simulation_tasks.append((sim, iter_num, iter_type, opt_type,
                                     pressure_threshold_mpa, max_penalty_percent))

    print(f"Processing {len(simulation_tasks)} simulations in parallel...")

    # Process simulations in parallel
    results = []
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(simulation_tasks))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_sim = {executor.submit(process_single_simulation, task): task
                         for task in simulation_tasks}

        completed_count = 0
        for future in as_completed(future_to_sim):
            result = future.result()
            if result is not None:
                results.append(result)
            completed_count += 1
            if completed_count % 100 == 0:
                print(f"  Processed {completed_count}/{len(simulation_tasks)} simulations")

    print(f"Completed parallel processing: {len(results)} valid results from {len(simulation_tasks)} simulations")
    return results


def diagnose_folder_structure(bo_folder, nsga3_folder):
    """Enhanced diagnostic function to check both pattern types."""
    folders_to_check = [
        ("BO", bo_folder),
        ("NSGA-III", nsga3_folder)
    ]

    for optimizer_name, folder_path in folders_to_check:
        print(f"\n{'=' * 60}")
        print(f"DIAGNOSING {optimizer_name} FOLDER")
        print(f"{'=' * 60}")
        print(f"Path: {folder_path}")

        if not os.path.exists(folder_path):
            print("Folder does not exist!")
            continue

        # Get all subfolders
        base_folder = Path(folder_path)
        subfolders = [f for f in base_folder.iterdir() if f.is_dir()]
        print(f"Found {len(subfolders)} subfolders")

        # Check first few subfolders to understand structure
        print(f"\nFirst 10 subfolders:")
        for i, folder in enumerate(subfolders[:10]):
            folder_name = folder.name
            print(f"  {i + 1:2d}. {folder_name}")

            # Check if this looks like iteration/generation folder
            if optimizer_name == "BO":
                if folder_name.startswith("Iteration_I") or folder_name == "Initial_Sampling":
                    print(f"      Valid {optimizer_name} iteration folder")
                else:
                    print(f"      Not a recognized {optimizer_name} iteration folder")
            else:  # NSGA-III
                if folder_name.startswith("Generation_G") or folder_name == "Initial_Sampling":
                    print(f"      Valid {optimizer_name} generation folder")
                else:
                    print(f"      Not a recognized {optimizer_name} generation folder")

        # Find a valid iteration/generation folder to examine simulation folders
        valid_iteration_folder = None
        for folder in subfolders:
            folder_name = folder.name
            if optimizer_name == "BO" and (folder_name.startswith("Iteration_I") or folder_name == "Initial_Sampling"):
                valid_iteration_folder = folder
                break
            elif optimizer_name == "NSGA-III" and (
                    folder_name.startswith("Generation_G") or folder_name == "Initial_Sampling"):
                valid_iteration_folder = folder
                break

        if valid_iteration_folder:
            print(f"\nExamining simulation folders in: {valid_iteration_folder.name}")
            sim_folders = [f for f in valid_iteration_folder.iterdir() if f.is_dir()]
            print(f"   Found {len(sim_folders)} simulation folders")

            print(f"\nPattern analysis for first 10 simulation folders:")
            new_pattern_matches = 0
            old_pattern_matches = 0

            for i, sim in enumerate(sim_folders[:10]):
                sim_name = sim.name
                print(f"     {i + 1:2d}. {sim_name}")

                # Check both patterns
                pattern_type, match = detect_folder_pattern(sim_name)

                if pattern_type == "new":
                    print(f"        MATCHES NEW pattern (with CL)")
                    new_pattern_matches += 1
                    params = {k: float(v) for k, v in match.groupdict().items()}
                    print(
                        f"        Parsed: CL={params['CL']}, dZ={params['dZ']}, LKG={params['LKG']}, lF={params['lF']}, zeta={params['zeta']}")
                elif pattern_type == "old":
                    print(f"        MATCHES OLD pattern (with dK)")
                    old_pattern_matches += 1
                    params = {k: float(v) for k, v in match.groupdict().items()}
                    print(
                        f"        Parsed: dK={params['dK']}, dZ={params['dZ']}, LKG={params['LKG']}, lF={params['lF']}, zeta={params['zeta']}")
                else:
                    print(f"        Does NOT match either pattern")

            print(f"\nPattern matching summary for {valid_iteration_folder.name}:")
            print(f"   - Total simulation folders: {len(sim_folders)}")
            print(f"   - Folders matching NEW pattern (CL): {new_pattern_matches}")
            print(f"   - Folders matching OLD pattern (dK): {old_pattern_matches}")
            total_matches = new_pattern_matches + old_pattern_matches
            print(f"   - Total matches: {total_matches}")
            print(f"   - Match rate: {total_matches / len(sim_folders) * 100:.1f}%" if len(
                sim_folders) > 0 else "   - Match rate: N/A")

            # Check file structure in one matching folder
            if total_matches > 0:
                matching_folder = None
                for sim in sim_folders:
                    pattern_type, _ = detect_folder_pattern(sim.name)
                    if pattern_type != "none":
                        matching_folder = sim
                        break

                if matching_folder:
                    print(f"\nChecking file structure in: {matching_folder.name}")

                    # Check for required files
                    required_files = [
                        'input/geometry.txt',
                        'input/operatingconditions.txt',
                        'output/piston/piston.txt',
                        'output/piston/matlab/Piston_Contact_Pressure.txt'
                    ]

                    for file_path in required_files:
                        full_path = matching_folder / file_path
                        exists = full_path.exists()
                        size = full_path.stat().st_size if exists else 0
                        print(
                            f"     {'Exists' if exists else 'Missing'} {file_path}: {f'({size} bytes)' if exists else ''}")
        else:
            print(f"\nNo valid iteration/generation folders found for {optimizer_name}")


def filter_data_by_phase(df_bo, df_nsga3, max_iteration):
    """
    Filter data by maximum iteration/generation number for phased plotting.

    Args:
        df_bo: BO DataFrame
        df_nsga3: NSGA-III DataFrame
        max_iteration: Maximum iteration/generation to include

    Returns:
        Filtered DataFrames
    """
    df_bo_filtered = df_bo.copy() if df_bo is not None and not df_bo.empty else pd.DataFrame()
    df_nsga3_filtered = df_nsga3.copy() if df_nsga3 is not None and not df_nsga3.empty else pd.DataFrame()

    # Filter BO data
    if not df_bo_filtered.empty and 'iteration' in df_bo_filtered.columns:
        df_bo_filtered = df_bo_filtered[df_bo_filtered['iteration'] <= max_iteration]

    # Filter NSGA-III data (using 'iteration' column after renaming)
    if not df_nsga3_filtered.empty and 'iteration' in df_nsga3_filtered.columns:
        df_nsga3_filtered = df_nsga3_filtered[df_nsga3_filtered['iteration'] <= max_iteration]

    return df_bo_filtered, df_nsga3_filtered


def create_plot_all_points_param_evolution_bo_nsga(df_bo, df_nsga3, output_dir, phase_suffix=""):
    """
    Parameter evolution plot showing ALL points for BO and NSGA-III with best-per-iteration
    lines and best-overall markers. Saves a PNG into output_dir.
    Returns True if created, else False.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Normalize inputs
    dfs = []
    labels = ["BO", "NSGA-III"]
    for df in [df_bo, df_nsga3]:
        if df is None or df.empty:
            dfs.append(pd.DataFrame());
            continue
        d = df.copy()
        if 'iteration' not in d.columns and 'generation' in d.columns:
            d = d.rename(columns={'generation': 'iteration'})
        if 'clearance' not in d.columns and {'dZ', 'dK'}.issubset(d.columns):
            d['clearance'] = d['dZ'] - d['dK']
        if 'iteration' in d.columns:
            d['iteration'] = pd.to_numeric(d['iteration'], errors='coerce')
        if not {'iteration', 'total'}.issubset(d.columns):
            dfs.append(pd.DataFrame());
            continue
        d = d.dropna(subset=['iteration', 'total'])
        dfs.append(d if not d.empty else pd.DataFrame())

    # If both empty, bail
    if all(d.empty for d in dfs):
        print("create_plot_all_points_param_evolution_bo_nsga: no data to plot.")
        return False

    # Best overall per DF
    best_overall = []
    for d in dfs:
        if not d.empty and 'total' in d.columns:
            best_overall.append(d.loc[d['total'].idxmin()])
        else:
            best_overall.append(None)

    # Parameters to plot
    parameters_to_plot = ["clearance", "LKG", "lF"]
    colors = ['blue', 'orange']
    markers = ['o', '^']

    fig, axes = plt.subplots(len(parameters_to_plot), 1, figsize=(6, 10), sharex=True)
    if len(parameters_to_plot) == 1:
        axes = [axes]
    handle_map = {}

    for i, param in enumerate(parameters_to_plot):
        ax = axes[i]
        for j, (d, lab) in enumerate(zip(dfs, labels)):
            if d.empty or param not in d.columns:
                continue
            dd = d.dropna(subset=[param]).copy()
            if dd.empty:
                continue

            jitter = np.random.normal(0, 0.15, size=len(dd))
            sc = ax.scatter(
                dd['iteration'] + jitter, dd[param],
                alpha=0.6, s=20, color=colors[j],
                label=f"{lab} All Points",
                edgecolors='none', marker=markers[j]
            )

            try:
                best_idx = dd.groupby('iteration', as_index=False)['total'].idxmin()
                idxs = best_idx.values if hasattr(best_idx, 'values') else list(best_idx)
                d_best = dd.loc[idxs]
            except Exception:
                d_best = dd.sort_values(['iteration', 'total']).drop_duplicates('iteration', keep='first')

            ln, = ax.plot(
                d_best['iteration'], d_best[param],
                marker=markers[j], linestyle='-', color=colors[j],
                label=f"{lab} Best per Iteration",
                linewidth=2, markersize=8,
                markeredgecolor='white', markeredgewidth=1
            )

            star = None
            bo = best_overall[j]
            if bo is not None and param in bo.index and pd.notna(bo['iteration']) and pd.notna(bo[param]):
                star = ax.scatter(
                    [bo['iteration']], [bo[param]],
                    s=200, color=('red' if j == 0 else 'darkred'),
                    marker='*', edgecolors='black', linewidth=2,
                    label=f"Best {lab} Overall", zorder=10
                )

            if i == 0:
                for h in [sc, ln] + ([star] if star is not None else []):
                    if h is not None:
                        labh = h.get_label()
                        if labh not in handle_map:
                            handle_map[labh] = h

        if param == "clearance":
            ylab = "clearance [μm]"
        elif param == "LKG":
            ylab = "LKG [mm]"
        elif param == "lF":
            ylab = "lF [mm]"
        else:
            ylab = param
        ax.set_ylabel(ylab, fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
    axes[-1].tick_params(axis='both', labelsize=16)

    handles = list(handle_map.values())
    labels = list(handle_map.keys())
    if handles:
        fig.legend(handles, labels, fontsize=12, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.02))

    title = f"Parameter Evolution: All Points vs Best Points (BO vs NSGA-III){phase_suffix}"
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    filename = f"param_evolution_allpoints_bo_nsga{phase_suffix.lower().replace(' ', '_').replace('-', '_')}.png"
    out_path = Path(output_dir) / filename
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Created plot: {out_path}")
    return True


def create_best_solutions_bar_plot_bo_nsga(df_bo, df_nsga3, output_path, phase_suffix=""):
    """Create a square bar plot comparing the best total loss for BO and NSGA-III."""
    import matplotlib.pyplot as plt
    import pandas as pd

    best_solutions = []
    technique_names = []

    # BO best solution
    if df_bo is not None and not df_bo.empty and 'total' in df_bo.columns:
        valid_bo = df_bo[(df_bo['total'] > 0) & (df_bo['total'] < 1e6)]
        if not valid_bo.empty:
            best_loss_bo = valid_bo['total'].min()
            best_solutions.append(best_loss_bo)
            technique_names.append("BO")

    # NSGA-III best solution
    if df_nsga3 is not None and not df_nsga3.empty and 'total' in df_nsga3.columns:
        valid_nsga3 = df_nsga3[(df_nsga3['total'] > 0) & (df_nsga3['total'] < 1e6)]
        if not valid_nsga3.empty:
            best_loss_nsga3 = valid_nsga3['total'].min()
            best_solutions.append(best_loss_nsga3)
            technique_names.append("NSGA-III")

    # Create bar plot
    if best_solutions:
        fig, ax = plt.subplots(figsize=(8, 8))  # Square plot

        # Colors for BO and NSGA-III
        colors = ['blue', 'orange']
        bars = ax.bar(technique_names, best_solutions,
                      color=colors[:len(technique_names)],
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Value labels on bars (normal formatting instead of scientific)
        for bar, value in zip(bars, best_solutions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.2f}',  # Changed from scientific notation to decimal
                    ha='center', va='bottom', fontsize=18, fontweight='bold')

        ax.set_xlabel('Optimization Method', fontsize=18)
        ax.set_ylabel('Best Total Loss [W]', fontsize=18)
        title = f'Best Solution Comparison: BO vs NSGA-III{phase_suffix}'
        ax.set_title(title, fontsize=16)

        # Removed logarithmic scale - now uses linear scale by default
        # ax.set_yscale('log')  # This line is removed

        ax.grid(True, alpha=0.3, axis='y')

        # Increased font size for tick labels
        plt.xticks(fontsize=20)  # Increased from 18 to 20
        plt.yticks(fontsize=20)  # Increased from 18 to 20

        plt.tight_layout()

        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created best solutions comparison bar plot: {output_path}")
        return True

    print("No valid BO or NSGA-III solutions found for bar plot.")
    return False

def create_convergence_plot_bo_nsga(df_bo, df_nsga3, output_dir, phase_suffix=""):
    """Create convergence plot showing best loss evolution over iterations."""

    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'orange']
    markers = ['o', '^']
    labels = ['BO', 'NSGA-III']

    for i, (df, label, color, marker) in enumerate(zip([df_bo, df_nsga3], labels, colors, markers)):
        if df is None or df.empty:
            continue

        d = df.copy()
        if 'iteration' not in d.columns and 'generation' in d.columns:
            d = d.rename(columns={'generation': 'iteration'})

        if not {'iteration', 'total'}.issubset(d.columns):
            continue

        d = d.dropna(subset=['iteration', 'total'])
        if d.empty:
            continue

        # Calculate best loss per iteration (cumulative minimum)
        d_sorted = d.sort_values('iteration')
        best_per_iter = d_sorted.groupby('iteration')['total'].min().reset_index()
        best_per_iter['cumulative_best'] = best_per_iter['total'].cummin()

        ax.plot(best_per_iter['iteration'], best_per_iter['cumulative_best'],
                marker=marker, linestyle='-', color=color, label=label,
                linewidth=2, markersize=6, markeredgecolor='white', markeredgewidth=1)

    ax.set_xlabel('Optimization Step (Iteration/Generation)', fontsize=14)
    ax.set_ylabel('Best Total Loss [W]', fontsize=14)
    title = f'Convergence: Best Loss Evolution{phase_suffix}'
    ax.set_title(title, fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    filename = f"convergence_bo_nsga{phase_suffix.lower().replace(' ', '_').replace('-', '_')}.png"
    out_path = Path(output_dir) / filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created convergence plot: {out_path}")
    return True


def create_phase_comparison_summary_plot(df_bo_phases, df_nsga3_phases, phase_names, output_dir):
    """Create a summary plot comparing best solutions across all phases."""

    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Best loss per phase
    bo_best_losses = []
    nsga3_best_losses = []

    for df_bo, df_nsga3 in zip(df_bo_phases, df_nsga3_phases):
        # BO best loss
        if df_bo is not None and not df_bo.empty and 'total' in df_bo.columns:
            valid_bo = df_bo[(df_bo['total'] > 0) & (df_bo['total'] < 1e6)]
            bo_best = valid_bo['total'].min() if not valid_bo.empty else None
        else:
            bo_best = None
        bo_best_losses.append(bo_best)

        # NSGA-III best loss
        if df_nsga3 is not None and not df_nsga3.empty and 'total' in df_nsga3.columns:
            valid_nsga3 = df_nsga3[(df_nsga3['total'] > 0) & (df_nsga3['total'] < 1e6)]
            nsga3_best = valid_nsga3['total'].min() if not valid_nsga3.empty else None
        else:
            nsga3_best = None
        nsga3_best_losses.append(nsga3_best)

    # Filter out None values for plotting
    x_pos = range(len(phase_names))
    bo_valid = [(i, loss) for i, loss in enumerate(bo_best_losses) if loss is not None]
    nsga3_valid = [(i, loss) for i, loss in enumerate(nsga3_best_losses) if loss is not None]

    if bo_valid:
        bo_x, bo_y = zip(*bo_valid)
        ax1.plot(bo_x, bo_y, 'o-', color='blue', label='BO', linewidth=2, markersize=8)

    if nsga3_valid:
        nsga3_x, nsga3_y = zip(*nsga3_valid)
        ax1.plot(nsga3_x, nsga3_y, '^-', color='orange', label='NSGA-III', linewidth=2, markersize=8)

    ax1.set_xlabel('Phase', fontsize=14)
    ax1.set_ylabel('Best Total Loss [W]', fontsize=14)
    ax1.set_title('Best Loss Evolution Across Phases', fontsize=16)
    ax1.set_yscale('log')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(phase_names)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Number of evaluations per phase
    bo_counts = [len(df) if df is not None and not df.empty else 0 for df in df_bo_phases]
    nsga3_counts = [len(df) if df is not None and not df.empty else 0 for df in df_nsga3_phases]

    width = 0.35
    x_pos = range(len(phase_names))

    ax2.bar([x - width / 2 for x in x_pos], bo_counts, width, label='BO', color='blue', alpha=0.7)
    ax2.bar([x + width / 2 for x in x_pos], nsga3_counts, width, label='NSGA-III', color='orange', alpha=0.7)

    ax2.set_xlabel('Phase', fontsize=14)
    ax2.set_ylabel('Number of Evaluations', fontsize=14)
    ax2.set_title('Evaluation Count per Phase', fontsize=16)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(phase_names)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    plt.tight_layout()

    out_path = Path(output_dir) / "phase_comparison_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created phase comparison summary plot: {out_path}")
    return True


# --- Main Orchestration and Utility Functions ---

def create_all_plots_for_phase(df_bo, df_nsga3, output_dir, phase_suffix, pressure_threshold_mpa=100.0):
    """Create all plots for a specific phase."""

    print(f"Creating plots for{phase_suffix}...")

    # Create output subdirectory for this phase
    phase_dir = Path(output_dir) / f"phase{phase_suffix.lower().replace(' ', '_').replace('-', '_')}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    plots_created = []

    # Parameter evolution plot
    if create_plot_all_points_param_evolution_bo_nsga(df_bo, df_nsga3, phase_dir, phase_suffix):
        plots_created.append("Parameter Evolution")

    # Best solutions bar plot
    bar_plot_path = phase_dir / f"best_solutions_bo_vs_nsga{phase_suffix.lower().replace(' ', '_').replace('-', '_')}.png"
    if create_best_solutions_bar_plot_bo_nsga(df_bo, df_nsga3, bar_plot_path, phase_suffix):
        plots_created.append("Best Solutions Bar Plot")

    # Convergence plot
    if create_convergence_plot_bo_nsga(df_bo, df_nsga3, phase_dir, phase_suffix):
        plots_created.append("Convergence Plot")

    print(f"  Created {len(plots_created)} plots for{phase_suffix}: {', '.join(plots_created)}")
    return len(plots_created)


def main_parallel():
    """Parallel version of the main function with phased plotting."""
    start_time = time.time()

    # Configuration
    bo_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\bayesian_optimization'
    nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\advanced_nsga3'

    pressure_threshold_mpa = 10.0
    max_penalty_percent = 200.0

    # Phase configuration - define the iteration/generation limits for each phase
    phase_configs = [
        {"name": "Phase 1", "max_iteration": 10, "suffix": " - Phase 1"},
        {"name": "Phase 1-2", "max_iteration": 20, "suffix": " - Phase 1-2"},
        {"name": "Phase 1-2-3", "max_iteration": 30, "suffix": " - Phase 1-2-3"}
    ]

    # Determine optimal number of workers
    num_workers = min(mp.cpu_count(), 24)
    print(f"Using {num_workers} parallel workers (out of {mp.cpu_count()} available CPUs)")

    print("=" * 80)
    print("PARALLEL OPTIMIZATION ANALYSIS WITH PHASED PLOTTING")
    print("=" * 80)
    print("Supported formats:")
    print("   - NEW: CL0.028_dZ19.888_LKG56.4_lF36.9_zeta5 (extracts dK from geometry.txt)")
    print("   - OLD: dK19.48_dZ19.51_LKG56.4_lF36.9_zeta5 (calculates clearance as dZ-dK)")
    print(f"Contact Pressure Penalty Settings:")
    print(f"   - Pressure Threshold: {pressure_threshold_mpa:.1f} MPa")
    print(f"   - Maximum Penalty: {max_penalty_percent:.0f}%")
    print(f"Phase Configuration:")
    for config in phase_configs:
        print(f"   - {config['name']}: Up to iteration/generation {config['max_iteration']}")
    print("=" * 80)

    # Run diagnostics first
    print("\nRUNNING DIAGNOSTICS...")
    diagnose_folder_structure(bo_folder, nsga3_folder)

    # Load all results in parallel (complete dataset)
    print("\nLOADING COMPLETE OPTIMIZATION RESULTS")
    print("=" * 60)
    load_start_time = time.time()

    with ProcessPoolExecutor(max_workers=2) as executor:
        future_bo = executor.submit(load_results_with_contact_pressure_parallel,
                                    bo_folder, "BO", pressure_threshold_mpa,
                                    max_penalty_percent, num_workers // 2)
        future_nsga3 = executor.submit(load_results_with_contact_pressure_parallel,
                                       nsga3_folder, "NSGA-III", pressure_threshold_mpa,
                                       max_penalty_percent, num_workers // 2)

        bo_results = future_bo.result()
        nsga3_results = future_nsga3.result()

    load_time = time.time() - load_start_time
    print(f"Data loading completed in {load_time:.2f} seconds")
    print(f"Loaded {len(bo_results)} valid BO results")
    print(f"Loaded {len(nsga3_results)} valid NSGA-III results")

    if len(bo_results) == 0 and len(nsga3_results) == 0:
        print("\nNO VALID RESULTS LOADED!")
        print("   Please check the diagnostic output above to identify the issue.")
        return

    # Convert to DataFrame
    df_bo_complete = pd.DataFrame(bo_results)
    df_nsga3_complete = pd.DataFrame(nsga3_results)

    # Ensure common iteration column
    if not df_nsga3_complete.empty and 'generation' in df_nsga3_complete.columns:
        df_nsga3_complete = df_nsga3_complete.rename(columns={'generation': 'iteration'})

    # Create output directory
    output_dir = Path("optimization_plots_phased")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print data summary
    print(f"\nCOMPLETE DATA SUMMARY:")
    if not df_bo_complete.empty:
        print(f"   BO data shape: {df_bo_complete.shape}")
        if 'iteration' in df_bo_complete.columns:
            print(
                f"   BO iteration range: {df_bo_complete['iteration'].min():.0f} - {df_bo_complete['iteration'].max():.0f}")

    if not df_nsga3_complete.empty:
        print(f"   NSGA-III data shape: {df_nsga3_complete.shape}")
        if 'iteration' in df_nsga3_complete.columns:
            print(
                f"   NSGA-III iteration range: {df_nsga3_complete['iteration'].min():.0f} - {df_nsga3_complete['iteration'].max():.0f}")

    # Create plots for each phase
    print("\nCREATING PHASED PLOTS")
    print("=" * 60)

    df_bo_phases = []
    df_nsga3_phases = []

    for i, phase_config in enumerate(phase_configs):
        print(f"\nProcessing {phase_config['name']} (up to iteration {phase_config['max_iteration']})...")

        # Filter data for this phase
        df_bo_phase, df_nsga3_phase = filter_data_by_phase(
            df_bo_complete, df_nsga3_complete, phase_config['max_iteration']
        )

        df_bo_phases.append(df_bo_phase)
        df_nsga3_phases.append(df_nsga3_phase)

        print(f"   BO data for {phase_config['name']}: {len(df_bo_phase)} samples")
        print(f"   NSGA-III data for {phase_config['name']}: {len(df_nsga3_phase)} samples")

        # Create plots for this phase
        plots_count = create_all_plots_for_phase(
            df_bo_phase, df_nsga3_phase, output_dir,
            phase_config['suffix'], pressure_threshold_mpa
        )

    # Create phase comparison summary
    print(f"\nCreating phase comparison summary...")
    phase_names = [config['name'] for config in phase_configs]
    create_phase_comparison_summary_plot(df_bo_phases, df_nsga3_phases, phase_names, output_dir)

    # Print final summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"PHASED PLOTTING COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Plots created for {len(phase_configs)} phases")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Phase comparison summary: {output_dir / 'phase_comparison_summary.png'}")

    # List all created plot directories
    print(f"\nPhase-specific plots created in:")
    for config in phase_configs:
        phase_dir_name = f"phase{config['suffix'].lower().replace(' ', '_').replace('-', '_')}"
        phase_dir = output_dir / phase_dir_name
        if phase_dir.exists():
            plot_files = list(phase_dir.glob("*.png"))
            print(f"   {phase_dir}: {len(plot_files)} plots")


if __name__ == "__main__":
    # Set start method for multiprocessing (important for Windows)
    mp.set_start_method('spawn', force=True)
    main_parallel()