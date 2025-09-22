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

# Import all functions from T1
import sys

sys.path.append('..')  # Add current directory to path
from T1 import (
    load_matlab_txt, round2, parse_geometry, parse_operating_conditions,
    create_cumulative_pressure_map, collect_simulation_data,
    create_comparison_plots, create_overall_summary, run_contact_pressure_analysis
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


# --- Basic Plotting Functions for Parallel Execution ---

def create_plot_convergence_bo(args):
    """Create BO convergence plot - for parallel execution."""
    df_bo, output_dir = args
    if df_bo.empty:
        return "convergence_BO.png", "No data"

    plt.figure(figsize=(8, 5))
    df_bo_best = df_bo.groupby('iteration')['total'].min().reset_index()
    plt.plot(df_bo_best['iteration'], df_bo_best['total'], marker='o', label='BO')
    plt.title("BO Convergence: Best Total Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Total Loss (min up to iteration)")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "convergence_BO.png")
    plt.close()
    return "convergence_BO.png", "Success"


def create_plot_convergence_nsga3(args):
    """Create NSGA-III convergence plot - for parallel execution."""
    df_nsga3, output_dir = args
    if df_nsga3.empty:
        return "convergence_NSGA3.png", "No data"

    plt.figure(figsize=(8, 5))
    df_nsga3_best = df_nsga3.groupby('iteration')['total'].min().reset_index()
    plt.plot(df_nsga3_best['iteration'], df_nsga3_best['total'], color='orange', marker='o', label='NSGA-III')
    plt.xlabel("Generation")
    plt.ylabel("Best Total Loss [W]")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "convergence_NSGA3.png")
    plt.close()
    return "convergence_NSGA3.png", "Success"


def create_plot_param_distribution_bo(args):
    """Create BO parameter distribution plots - for parallel execution."""
    df_bo, output_dir, parameters = args
    if df_bo.empty:
        return "param_distribution_BO.png", "No data"

    fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
    for i, param in enumerate(parameters):
        sns.histplot(df_bo, x=param, kde=True, ax=axes[i], color='blue', bins=20)
        axes[i].set_title(f"BO: Distribution of {param}")
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "param_distribution_BO.png")
    plt.close(fig)
    return "param_distribution_BO.png", "Success"


def create_plot_param_distribution_nsga3(args):
    """Create NSGA-III parameter distribution plots - for parallel execution."""
    df_nsga3, output_dir, parameters = args
    if df_nsga3.empty:
        return "param_distribution_NSGA3.png", "No data"

    fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
    for i, param in enumerate(parameters):
        sns.histplot(df_nsga3, x=param, kde=True, ax=axes[i], color='orange', bins=20)
        axes[i].set_title(f"NSGA-III: Distribution of {param}")
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "param_distribution_NSGA3.png")
    plt.close(fig)
    return "param_distribution_NSGA3.png", "Success"


def create_plot_param_vs_loss_bo(args):
    """Create BO parameter vs loss scatter plots - for parallel execution."""
    df_bo, output_dir, parameters = args
    if df_bo.empty:
        return "param_vs_loss_BO.png", "No data"

    fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
    for i, param in enumerate(parameters):
        axes[i].scatter(df_bo[param], df_bo["total"], color='blue', alpha=0.6)
        axes[i].set_title(f"BO: {param} vs Total Loss")
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Total Loss")
        axes[i].set_yscale('log')
        axes[i].grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "param_vs_loss_BO.png")
    plt.close(fig)
    return "param_vs_loss_BO.png", "Success"


def create_plot_param_vs_loss_nsga3(args):
    """Create NSGA-III parameter vs loss scatter plots - for parallel execution."""
    df_nsga3, output_dir, parameters = args
    if df_nsga3.empty:
        return "param_vs_loss_NSGA3.png", "No data"

    fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
    for i, param in enumerate(parameters):
        axes[i].scatter(df_nsga3[param], df_nsga3["total"], color='orange', alpha=0.6)
        axes[i].set_title(f"NSGA-III: {param} vs Total Loss")
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Total Loss")
        axes[i].set_yscale('log')
        axes[i].grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "param_vs_loss_NSGA3.png")
    plt.close(fig)
    return "param_vs_loss_NSGA3.png", "Success"


def create_plot_pairplot_bo(args):
    """Create BO pairplot - for parallel execution."""
    df_bo, output_dir, parameters = args
    if df_bo.empty:
        return "pairplot_BO.png", "No data"

    cols_for_pairplot = parameters + ["total"]
    df_plot = df_bo.copy()
    if len(df_plot) > 5000:
        df_plot = df_plot.sample(n=5000, random_state=42)
    sns.pairplot(df_plot[cols_for_pairplot], diag_kind="kde")
    plt.suptitle("BO Pairwise Parameter Relationships", y=1.02)
    plt.savefig(output_dir / "pairplot_BO.png")
    plt.close()
    return "pairplot_BO.png", "Success"


def create_plot_pairplot_nsga3(args):
    """Create NSGA-III pairplot - for parallel execution."""
    df_nsga3, output_dir, parameters = args
    if df_nsga3.empty:
        return "pairplot_NSGA3.png", "No data"

    cols_for_pairplot = parameters + ["total"]
    df_plot = df_nsga3.copy()
    if len(df_plot) > 5000:
        df_plot = df_plot.sample(n=5000, random_state=42)
    sns.pairplot(df_plot[cols_for_pairplot], diag_kind="kde")
    plt.suptitle("NSGA-III Pairwise Parameter Relationships", y=1.02)
    plt.savefig(output_dir / "pairplot_NSGA3.png")
    plt.close()
    return "pairplot_NSGA3.png", "Success"


def create_plot_pareto_bo(args):
    """Create BO Pareto front plot - for parallel execution."""
    df_bo, output_dir = args
    if df_bo.empty or "mechanical" not in df_bo.columns or "volumetric" not in df_bo.columns:
        return "pareto_BO.png", "No data or missing columns"

    plt.figure(figsize=(6, 5))
    plt.scatter(df_bo["mechanical"], df_bo["volumetric"], color='blue', alpha=0.7)
    plt.title("BO: Mechanical vs Volumetric Loss")
    plt.xlabel("Mechanical Loss")
    plt.ylabel("Volumetric Loss")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_dir / "pareto_BO.png")
    plt.close()
    return "pareto_BO.png", "Success"


def create_plot_pareto_nsga3(args):
    """Create NSGA-III Pareto front plot - for parallel execution."""
    df_nsga3, output_dir = args
    if df_nsga3.empty or "mechanical" not in df_nsga3.columns or "volumetric" not in df_nsga3.columns:
        return "pareto_NSGA3.png", "No data or missing columns"

    plt.figure(figsize=(6, 5))
    plt.scatter(df_nsga3["mechanical"], df_nsga3["volumetric"], color='orange', alpha=0.7)
    plt.title("NSGA-III: Mechanical vs Volumetric Loss")
    plt.xlabel("Mechanical Loss")
    plt.ylabel("Volumetric Loss")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(output_dir / "pareto_NSGA3.png")
    plt.close()
    return "pareto_NSGA3.png", "Success"


# --- Advanced Combined Plotting Functions ---

def create_plot_combined_param_evolution_with_clearance(args):
    """Create combined parameter evolution plot with clearance and convergence - for parallel execution."""
    df_bo, df_nsga3, output_dir = args

    # Prepare best iteration data
    df_bo_best_iter_combined = pd.DataFrame()
    if not df_bo.empty:
        df_bo_best_iter_combined = df_bo.loc[df_bo.groupby('iteration')['total'].idxmin()].copy()
        df_bo_best_iter_combined['clearance'] = df_bo_best_iter_combined['dZ'] - df_bo_best_iter_combined['dK']

    df_nsga3_best_iter_combined = pd.DataFrame()
    if not df_nsga3.empty:
        df_nsga3_best_iter_combined = df_nsga3.loc[df_nsga3.groupby('iteration')['total'].idxmin()].copy()
        df_nsga3_best_iter_combined['clearance'] = df_nsga3_best_iter_combined['dZ'] - df_nsga3_best_iter_combined['dK']

    parameters_with_clearance = ["clearance", "LKG", "lF", "zeta"]
    plots_to_show = parameters_with_clearance + ["convergence"]

    if not df_bo_best_iter_combined.empty or not df_nsga3_best_iter_combined.empty:
        fig, axes = plt.subplots(len(plots_to_show), 1, figsize=(10, 18), sharex=True)
        if len(plots_to_show) == 1:
            axes = [axes]

        for i, param in enumerate(plots_to_show):
            if param == "convergence":
                # Special handling for convergence plot
                if not df_bo.empty:
                    df_bo_conv = df_bo.groupby('iteration')['total'].min().reset_index()
                    axes[i].plot(df_bo_conv['iteration'], df_bo_conv['total'],
                                 marker='o', linestyle='-', color='blue', label='BO', linewidth=2)

                if not df_nsga3.empty:
                    df_nsga3_conv = df_nsga3.groupby('iteration')['total'].min().reset_index()
                    axes[i].plot(df_nsga3_conv['iteration'], df_nsga3_conv['total'],
                                 marker='x', linestyle='--', color='orange', label='NSGA-III', linewidth=2)

                axes[i].set_ylabel("Best Total Power Loss [W]", fontsize=16)
                axes[i].set_yscale('log')
            else:
                # Regular parameter plots
                if not df_bo_best_iter_combined.empty:
                    axes[i].plot(df_bo_best_iter_combined['iteration'], df_bo_best_iter_combined[param],
                                 marker='o', linestyle='-', color='blue', label='BO')
                if not df_nsga3_best_iter_combined.empty:
                    axes[i].plot(df_nsga3_best_iter_combined['iteration'], df_nsga3_best_iter_combined[param],
                                 marker='x', linestyle='--', color='orange', label='NSGA-III')

                # Set proper labels for parameters
                if param == "zeta":
                    y_label = "γ"
                elif param == "clearance":
                    y_label = "clearance [um]"
                elif param == "LKG":
                    y_label = "LKG [mm]"
                elif param == "lF":
                    y_label = "lF [mm]"
                else:
                    y_label = param

                axes[i].set_ylabel(y_label, fontsize=16)

            axes[i].tick_params(axis='both', labelsize=16)
            axes[i].grid(True)
            axes[i].legend(fontsize=18)

        axes[-1].set_xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
        axes[-1].tick_params(axis='both', labelsize=16)

        fig.tight_layout()
        fig.savefig(output_dir / "combined_param_evolution_with_clearance_and_convergence.png", dpi=300)
        plt.close(fig)
        return "combined_param_evolution_with_clearance_and_convergence.png", "Success"

    return "combined_param_evolution_with_clearance_and_convergence.png", "No data"


def create_plot_all_points_param_evolution_with_clearance(args):
    """Create parameter evolution plot showing ALL points for each iteration/generation - for parallel execution."""
    df_bo, df_nsga3, output_dir = args

    # Add clearance calculation
    if not df_bo.empty:
        df_bo = df_bo.copy()
        df_bo['clearance'] = df_bo['dZ'] - df_bo['dK']
    if not df_nsga3.empty:
        df_nsga3 = df_nsga3.copy()
        df_nsga3['clearance'] = df_nsga3['dZ'] - df_nsga3['dK']

    # Find best overall solutions (across all iterations/generations)
    best_bo_overall = None
    best_nsga3_overall = None

    if not df_bo.empty:
        best_bo_overall_idx = df_bo['total'].idxmin()
        best_bo_overall = df_bo.loc[best_bo_overall_idx]

    if not df_nsga3.empty:
        best_nsga3_overall_idx = df_nsga3['total'].idxmin()
        best_nsga3_overall = df_nsga3.loc[best_nsga3_overall_idx]

    parameters_with_clearance = ["clearance", "LKG", "lF", "zeta"]
    plots_to_show = parameters_with_clearance + ["convergence"]

    if not df_bo.empty or not df_nsga3.empty:
        fig, axes = plt.subplots(len(plots_to_show), 1, figsize=(12, 20), sharex=True)
        if len(plots_to_show) == 1:
            axes = [axes]

        for i, param in enumerate(plots_to_show):
            if param == "convergence":
                # Special handling for convergence plot (keep as line plot with best values)
                if not df_bo.empty:
                    df_bo_conv = df_bo.groupby('iteration')['total'].min().reset_index()
                    axes[i].plot(df_bo_conv['iteration'], df_bo_conv['total'],
                                 marker='o', linestyle='-', color='blue', label='BO Best', linewidth=2, markersize=6)

                if not df_nsga3.empty:
                    df_nsga3_conv = df_nsga3.groupby('iteration')['total'].min().reset_index()
                    axes[i].plot(df_nsga3_conv['iteration'], df_nsga3_conv['total'],
                                 marker='x', linestyle='-', color='orange', label='NSGA-III Best', linewidth=2,
                                 markersize=8)

                # Add markers for best overall solutions on convergence plot
                if best_bo_overall is not None:
                    axes[i].scatter([best_bo_overall['iteration']], [best_bo_overall['total']],
                                    s=200, color='red', marker='*', edgecolors='black', linewidth=2,
                                    label='Best BO Overall', zorder=10)

                if best_nsga3_overall is not None:
                    axes[i].scatter([best_nsga3_overall['iteration']], [best_nsga3_overall['total']],
                                    s=200, color='darkred', marker='*', edgecolors='black', linewidth=2,
                                    label='Best NSGA-III Overall', zorder=10)

                axes[i].set_ylabel("Best Total Power Loss [W]", fontsize=14)
                axes[i].set_yscale('log')
            else:
                # Scatter plot showing ALL points for each iteration/generation
                if not df_bo.empty and param in df_bo.columns:
                    # Add some jitter to iteration numbers for better visibility
                    jitter_bo = np.random.normal(0, 0.15, len(df_bo))
                    axes[i].scatter(df_bo['iteration'] + jitter_bo, df_bo[param],
                                    alpha=0.6, s=20, color='blue', label='BO All Points', edgecolors='none')

                    # Overlay the best points for each iteration
                    df_bo_best = df_bo.loc[df_bo.groupby('iteration')['total'].idxmin()]
                    axes[i].plot(df_bo_best['iteration'], df_bo_best[param],
                                 marker='o', linestyle='-', color='darkblue', label='BO Best per Iteration',
                                 linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1)

                    # Add marker for best overall BO solution
                    if best_bo_overall is not None:
                        axes[i].scatter([best_bo_overall['iteration']], [best_bo_overall[param]],
                                        s=200, color='red', marker='*', edgecolors='black', linewidth=2,
                                        label='Best BO Overall', zorder=10)

                if not df_nsga3.empty and param in df_nsga3.columns:
                    # Add some jitter to iteration numbers for better visibility
                    jitter_nsga3 = np.random.normal(0, 0.15, len(df_nsga3))
                    axes[i].scatter(df_nsga3['iteration'] + jitter_nsga3, df_nsga3[param],
                                    alpha=0.6, s=20, color='orange', label='NSGA-III All Points',
                                    edgecolors='none', marker='^')

                    # Overlay the best points for each iteration
                    df_nsga3_best = df_nsga3.loc[df_nsga3.groupby('iteration')['total'].idxmin()]
                    axes[i].plot(df_nsga3_best['iteration'], df_nsga3_best[param],
                                 marker='^', linestyle='-', color='darkorange', label='NSGA-III Best per Generation',
                                 linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1)

                    # Add marker for best overall NSGA-III solution
                    if best_nsga3_overall is not None:
                        axes[i].scatter([best_nsga3_overall['iteration']], [best_nsga3_overall[param]],
                                        s=200, color='darkred', marker='*', edgecolors='black', linewidth=2,
                                        label='Best NSGA-III Overall', zorder=10)

                # Set proper labels for parameters
                if param == "zeta":
                    y_label = "γ"
                elif param == "clearance":
                    y_label = "clearance [μm]"
                elif param == "LKG":
                    y_label = "LKG [mm]"
                elif param == "lF":
                    y_label = "lF [mm]"
                else:
                    y_label = param

                axes[i].set_ylabel(y_label, fontsize=14)

            axes[i].tick_params(axis='both', labelsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=10, loc='best')

        axes[-1].set_xlabel("Optimization Step (Iteration/Generation)", fontsize=14)
        axes[-1].tick_params(axis='both', labelsize=12)

        fig.suptitle("Parameter Evolution: All Selected Points vs Best Points", fontsize=16, y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / "all_points_param_evolution_with_clearance_and_convergence.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        return "all_points_param_evolution_with_clearance_and_convergence.png", "Success"

    return "all_points_param_evolution_with_clearance_and_convergence.png", "No data"


def create_plot_combined_param_evolution_original(args):
    """Create original combined parameter evolution plot - for parallel execution."""
    df_bo, df_nsga3, output_dir = args
    parameters = ["dK", "dZ", "LKG", "lF", "zeta"]

    df_bo_best_iter_combined = pd.DataFrame()
    if not df_bo.empty:
        df_bo_best_iter_combined = df_bo.loc[df_bo.groupby('iteration')['total'].idxmin()]

    df_nsga3_best_iter_combined = pd.DataFrame()
    if not df_nsga3.empty:
        df_nsga3_best_iter_combined = df_nsga3.loc[df_nsga3.groupby('iteration')['total'].idxmin()]

    if not df_bo_best_iter_combined.empty or not df_nsga3_best_iter_combined.empty:
        fig, axes = plt.subplots(len(parameters), 1, figsize=(10, 15), sharex=True)
        if len(parameters) == 1:
            axes = [axes]

        for i, param in enumerate(parameters):
            if not df_bo_best_iter_combined.empty:
                axes[i].plot(df_bo_best_iter_combined['iteration'], df_bo_best_iter_combined[param],
                             marker='o', linestyle='-', color='blue', label='BO')
            if not df_nsga3_best_iter_combined.empty:
                axes[i].plot(df_nsga3_best_iter_combined['iteration'], df_nsga3_best_iter_combined[param],
                             marker='x', linestyle='--', color='orange', label='NSGA-III')

            y_label = "γ" if param == "zeta" else param
            axes[i].set_ylabel(y_label, fontsize=16)

            # Calculate y-limits
            all_param_values = []
            if not df_bo_best_iter_combined.empty:
                all_param_values.extend(df_bo_best_iter_combined[param].tolist())
            if not df_nsga3_best_iter_combined.empty:
                all_param_values.extend(df_nsga3_best_iter_combined[param].tolist())
            if all_param_values:
                min_val = min(all_param_values)
                max_val = max(all_param_values)
                if min_val == max_val:
                    pad = 0.1 * abs(min_val) if min_val != 0 else 0.1
                    axes[i].set_ylim(min_val - pad, max_val + pad)
                else:
                    range_val = max_val - min_val
                    axes[i].set_ylim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)

            axes[i].tick_params(axis='both', labelsize=16)
            axes[i].grid(True)
            axes[i].legend(fontsize=18)

        axes[-1].set_xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
        axes[-1].tick_params(axis='both', labelsize=16)

        fig.tight_layout()
        fig.savefig(output_dir / "combined_param_evolution.png")
        plt.close(fig)
        return "combined_param_evolution.png", "Success"

    return "combined_param_evolution.png", "No data"


def create_plot_combined_convergence(args):
    """Create combined convergence plot - for parallel execution."""
    df_bo, df_nsga3, output_dir = args

    plt.figure(figsize=(9, 6))
    plot_data_exists = False

    if not df_bo.empty:
        df_bo_best_conv = df_bo.groupby('iteration')['total'].min().reset_index()
        plt.plot(df_bo_best_conv['iteration'], df_bo_best_conv['total'], marker='o', linestyle='-', color='blue',
                 label='BO')
        plot_data_exists = True

    if not df_nsga3.empty:
        df_nsga3_best_conv = df_nsga3.groupby('iteration')['total'].min().reset_index()
        plt.plot(df_nsga3_best_conv['iteration'], df_nsga3_best_conv['total'], marker='x', linestyle='--',
                 color='orange', label='NSGA-III')
        plot_data_exists = True

    if plot_data_exists:
        plt.xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
        plt.ylabel("Best Total Loss [W]", fontsize=16)
        plt.yscale('log')
        plt.grid(True)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / "combined_convergence.png")
        plt.close()
        return "combined_convergence.png", "Success"

    return "combined_convergence.png", "No data"


def create_plot_combined_pareto(args):
    """Create combined Pareto front plot - for parallel execution."""
    df_bo, df_nsga3, output_dir = args

    if not df_bo.empty or not df_nsga3.empty:
        plt.figure(figsize=(7, 6))

        if not df_bo.empty and "mechanical" in df_bo.columns and "volumetric" in df_bo.columns:
            plt.scatter(df_bo["mechanical"], df_bo["volumetric"], color='blue', alpha=0.6, label='BO')

        if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
            plt.scatter(df_nsga3["mechanical"], df_nsga3["volumetric"], color='orange', alpha=0.6, label='NSGA-III')

        plt.xlabel("Mechanical Loss[W]", fontsize=16)
        plt.ylabel("Volumetric Loss[W]", fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(output_dir / "combined_pareto.png")
        plt.close()
        return "combined_pareto.png", "Success"

    return "combined_pareto.png", "No data"


# --- Enhanced Analysis and Contact Pressure Functions ---

def create_plot_contact_pressure_vs_loss_analysis(args):
    """Create contact pressure vs total loss analysis - for parallel execution."""
    df_bo, df_nsga3, output_dir, pressure_threshold_mpa = args

    # Find best solutions
    best_bo_penalty_idx = None
    best_nsga3_penalty_idx = None
    best_bo_no_penalty_idx = None
    best_nsga3_no_penalty_idx = None

    if not df_bo.empty:
        best_bo_penalty_idx = df_bo['penalized_total'].idxmin()
        best_bo_no_penalty_idx = df_bo['total'].idxmin()

    if not df_nsga3.empty:
        best_nsga3_penalty_idx = df_nsga3['penalized_total'].idxmin()
        best_nsga3_no_penalty_idx = df_nsga3['total'].idxmin()

    plt.figure(figsize=(12, 8))

    # Create 2x2 subplot for comprehensive view
    plt.subplot(2, 2, 1)
    if not df_bo.empty:
        scatter = plt.scatter(df_bo["total"], df_bo["max_contact_pressure"],
                              c=df_bo["contact_pressure_penalty"] * 100,
                              alpha=0.7, s=50, cmap='viridis', label='BO')
        plt.colorbar(scatter, label='Penalty (%)')

    plt.xlabel("Total Loss [W]", fontsize=12)
    plt.ylabel("Max Contact Pressure [MPa]", fontsize=12)
    plt.title("BO: Contact Pressure vs Total Loss", fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=pressure_threshold_mpa, color='red', linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 2)
    if not df_nsga3.empty:
        scatter = plt.scatter(df_nsga3["total"], df_nsga3["max_contact_pressure"],
                              c=df_nsga3["contact_pressure_penalty"] * 100,
                              alpha=0.7, s=50, cmap='viridis', label='NSGA-III')
        plt.colorbar(scatter, label='Penalty (%)')

    plt.xlabel("Total Loss [W]", fontsize=12)
    plt.ylabel("Max Contact Pressure [MPa]", fontsize=12)
    plt.title("NSGA-III: Contact Pressure vs Total Loss", fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=pressure_threshold_mpa, color='red', linestyle='--', alpha=0.7)

    # Combined plot
    plt.subplot(2, 1, 2)
    if not df_bo.empty:
        plt.scatter(df_bo["total"], df_bo["max_contact_pressure"],
                    color='blue', alpha=0.6, s=30, label='BO')

        # Highlight best solutions
        if best_bo_penalty_idx is not None:
            best_bo_solution = df_bo.loc[best_bo_penalty_idx]
            plt.scatter([best_bo_solution["total"]], [best_bo_solution["max_contact_pressure"]],
                        color='red', s=200, marker='*', edgecolors='black', linewidth=2,
                        label=f'Best BO (with penalty)', zorder=5)

        if best_bo_no_penalty_idx is not None and best_bo_no_penalty_idx != best_bo_penalty_idx:
            best_bo_no_penalty = df_bo.loc[best_bo_no_penalty_idx]
            plt.scatter([best_bo_no_penalty["total"]], [best_bo_no_penalty["max_contact_pressure"]],
                        color='lightcoral', s=150, marker='o', edgecolors='black', linewidth=2,
                        label=f'Best BO (no penalty)', zorder=4)

    if not df_nsga3.empty:
        plt.scatter(df_nsga3["total"], df_nsga3["max_contact_pressure"],
                    color='orange', alpha=0.6, s=30, label='NSGA-III')

        # Highlight best solutions
        if best_nsga3_penalty_idx is not None:
            best_nsga3_solution = df_nsga3.loc[best_nsga3_penalty_idx]
            plt.scatter([best_nsga3_solution["total"]], [best_nsga3_solution["max_contact_pressure"]],
                        color='darkred', s=200, marker='*', edgecolors='black', linewidth=2,
                        label=f'Best NSGA-III (with penalty)', zorder=5)

        if best_nsga3_no_penalty_idx is not None and best_nsga3_no_penalty_idx != best_nsga3_penalty_idx:
            best_nsga3_no_penalty = df_nsga3.loc[best_nsga3_no_penalty_idx]
            plt.scatter([best_nsga3_no_penalty["total"]], [best_nsga3_no_penalty["max_contact_pressure"]],
                        color='navajowhite', s=150, marker='^', edgecolors='black', linewidth=2,
                        label=f'Best NSGA-III (no penalty)', zorder=4)

    plt.axhline(y=pressure_threshold_mpa, color='red', linestyle='--',
                label=f'Pressure Threshold ({pressure_threshold_mpa} MPa)', linewidth=2)

    plt.xlabel("Total Loss [W]", fontsize=14)
    plt.ylabel("Max Contact Pressure [MPa]", fontsize=14)
    plt.title("Contact Pressure vs Total Loss - Best Solutions Comparison", fontsize=16)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "contact_pressure_vs_loss_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    return "contact_pressure_vs_loss_analysis.png", "Success"


def create_enhanced_combined_plots_with_penalty_parallel(df_bo, df_nsga3, output_dir, pressure_threshold_mpa=100.0):
    """Enhanced plots with parallel data preparation."""

    def find_best_solutions(df, penalty_column='penalized_total', loss_column='total'):
        if df.empty:
            return None, None, None, None
        best_penalty_idx = df[penalty_column].idxmin()
        best_loss_idx = df[loss_column].idxmin()
        return (best_penalty_idx, df.loc[best_penalty_idx],
                best_loss_idx, df.loc[best_loss_idx])

    def prepare_convergence_data(df, group_col, value_col):
        if df.empty:
            return pd.DataFrame()
        return df.groupby(group_col)[value_col].min().reset_index()

    # Parallel data preparation
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Find best solutions
        future_bo_best = executor.submit(find_best_solutions, df_bo)
        future_nsga3_best = executor.submit(find_best_solutions, df_nsga3)

        # Prepare convergence data
        future_bo_conv = executor.submit(prepare_convergence_data, df_bo, 'iteration', 'total')
        future_nsga3_conv = executor.submit(prepare_convergence_data, df_nsga3, 'iteration', 'total')
        future_bo_conv_penalty = executor.submit(prepare_convergence_data, df_bo, 'iteration', 'penalized_total')
        future_nsga3_conv_penalty = executor.submit(prepare_convergence_data, df_nsga3, 'iteration', 'penalized_total')
        future_bo_pressure = executor.submit(prepare_convergence_data, df_bo, 'iteration', 'max_contact_pressure')
        future_nsga3_pressure = executor.submit(prepare_convergence_data, df_nsga3, 'iteration', 'max_contact_pressure')

        # Collect results
        bo_results = future_bo_best.result()
        nsga3_results = future_nsga3_best.result()
        df_bo_conv = future_bo_conv.result()
        df_nsga3_conv = future_nsga3_conv.result()
        df_bo_conv_penalty = future_bo_conv_penalty.result()
        df_nsga3_conv_penalty = future_nsga3_conv_penalty.result()
        df_bo_pressure = future_bo_pressure.result()
        df_nsga3_pressure = future_nsga3_pressure.result()

    # Extract best solutions
    best_bo_solution = bo_results[1] if bo_results[0] is not None else None
    best_nsga3_solution = nsga3_results[1] if nsga3_results[0] is not None else None
    best_bo_no_penalty = bo_results[3] if bo_results[0] is not None else None
    best_nsga3_no_penalty = nsga3_results[3] if nsga3_results[0] is not None else None

    # Print best solutions
    if best_bo_solution is not None:
        print(f"Best BO solution (with penalty): Penalized Loss = {best_bo_solution['penalized_total']:.2e} W, "
              f"Contact Pressure = {best_bo_solution['max_contact_pressure']:.1f} MPa")
    if best_nsga3_solution is not None:
        print(
            f"Best NSGA-III solution (with penalty): Penalized Loss = {best_nsga3_solution['penalized_total']:.2e} W, "
            f"Contact Pressure = {best_nsga3_solution['max_contact_pressure']:.1f} MPa")

    # Create the enhanced convergence plot
    plt.figure(figsize=(12, 8))
    plot_data_exists = False

    if not df_bo_conv.empty:
        plt.subplot(2, 1, 1)
        plt.plot(df_bo_conv['iteration'], df_bo_conv['total'],
                 marker='o', linestyle='-', color='blue', label='BO (Total Loss)', linewidth=2, markersize=6)

        if not df_bo_conv_penalty.empty:
            plt.plot(df_bo_conv_penalty['iteration'], df_bo_conv_penalty['penalized_total'],
                     marker='s', linestyle='--', color='darkblue', label='BO (Penalized)', linewidth=2, markersize=6)
        plot_data_exists = True

    if not df_nsga3_conv.empty:
        if not plot_data_exists:
            plt.subplot(2, 1, 1)
        plt.plot(df_nsga3_conv['iteration'], df_nsga3_conv['total'],
                 marker='x', linestyle='-', color='orange', label='NSGA-III (Total Loss)', linewidth=2, markersize=8)

        if not df_nsga3_conv_penalty.empty:
            plt.plot(df_nsga3_conv_penalty['iteration'], df_nsga3_conv_penalty['penalized_total'],
                     marker='^', linestyle='--', color='darkorange', label='NSGA-III (Penalized)', linewidth=2,
                     markersize=8)
        plot_data_exists = True

    if plot_data_exists:
        plt.xlabel("Optimization Step (Iteration/Generation)", fontsize=14)
        plt.ylabel("Best Loss [W]", fontsize=14)
        plt.title("Optimization Convergence: Total Loss vs Penalized Loss", fontsize=16)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Contact pressure subplot
        plt.subplot(2, 1, 2)

        if not df_bo_pressure.empty:
            plt.plot(df_bo_pressure['iteration'], df_bo_pressure['max_contact_pressure'],
                     marker='o', linestyle='-', color='blue', label='BO Min Pressure', linewidth=2, markersize=6)

        if not df_nsga3_pressure.empty:
            plt.plot(df_nsga3_pressure['iteration'], df_nsga3_pressure['max_contact_pressure'],
                     marker='x', linestyle='-', color='orange', label='NSGA-III Min Pressure', linewidth=2,
                     markersize=8)

        plt.axhline(y=pressure_threshold_mpa, color='red', linestyle=':',
                    label=f'Pressure Threshold ({pressure_threshold_mpa} MPa)', linewidth=2)
        plt.xlabel("Optimization Step (Iteration/Generation)", fontsize=14)
        plt.ylabel("Max Contact Pressure [MPa]", fontsize=14)
        plt.title("Contact Pressure Evolution", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / "enhanced_convergence_with_penalty.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Enhanced convergence plot with penalty saved")


def create_sensitivity_analysis_parallel(df_bo, df_nsga3, output_dir):
    """Create sensitivity analysis with parallel correlation calculations."""

    def calculate_correlations(df, parameters):
        """Calculate correlations for a single dataset."""
        correlations = {}
        for param in parameters:
            if not df.empty and param in df.columns:
                try:
                    corr, _ = pearsonr(df[param], df["total"])
                    correlations[param] = abs(corr)
                except:
                    correlations[param] = np.nan
            else:
                correlations[param] = np.nan
        return correlations

    # Add clearance calculation
    if not df_bo.empty:
        df_bo["clearance"] = df_bo["dZ"] - df_bo["dK"]
    if not df_nsga3.empty:
        df_nsga3["clearance"] = df_nsga3["dZ"] - df_nsga3["dK"]

    selected_params = ["clearance", "zeta", "lF", "LKG"]
    param_labels = {"clearance": "clearance", "zeta": "γ", "lF": "lF", "LKG": "LKG"}

    # Calculate correlations in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bo = executor.submit(calculate_correlations, df_bo, selected_params)
        future_nsga3 = executor.submit(calculate_correlations, df_nsga3, selected_params)

        bo_correlations = future_bo.result()
        nsga3_correlations = future_nsga3.result()

    # Prepare data for plotting
    sensitivity_data = []
    for param in selected_params:
        sensitivity_data.append({
            "parameter": param_labels[param],
            "BO": bo_correlations[param],
            "NSGA-III": nsga3_correlations[param]
        })

    df_sensitivity = pd.DataFrame(sensitivity_data)

    # Create plot
    x = np.arange(len(df_sensitivity["parameter"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, df_sensitivity["BO"], width, label='BO', color='blue')
    bars2 = ax.bar(x + width / 2, df_sensitivity["NSGA-III"], width, label='NSGA-III', color='orange')

    ax.set_xlabel("Parameter", fontsize=16)
    ax.set_ylabel("Sensitivity (|Pearson Correlation|)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sensitivity["parameter"], fontsize=16)
    ax.legend(fontsize=18)
    ax.grid(True, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_bar_chart_all_params.png")
    plt.close(fig)
    return "sensitivity_bar_chart_all_params.png"


def create_penalty_distribution_analysis_parallel(df_bo, df_nsga3, output_dir, pressure_threshold_mpa=100.0):
    """Create penalty distribution analysis with parallel data filtering."""

    def filter_valid_data(df):
        """Filter for valid contact pressure data."""
        if df.empty:
            return pd.DataFrame()
        return df[df['contact_pressure_valid'] == True]

    # Filter data in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bo = executor.submit(filter_valid_data, df_bo)
        future_nsga3 = executor.submit(filter_valid_data, df_nsga3)

        valid_bo = future_bo.result()
        valid_nsga3 = future_nsga3.result()

    if valid_bo.empty and valid_nsga3.empty:
        print("No valid contact pressure data for penalty analysis")
        return

    plt.figure(figsize=(12, 8))

    # Penalty distribution
    plt.subplot(2, 2, 1)
    if not valid_bo.empty:
        plt.hist(valid_bo['contact_pressure_penalty'] * 100, bins=20, alpha=0.7, color='blue', label='BO')
    if not valid_nsga3.empty:
        plt.hist(valid_nsga3['contact_pressure_penalty'] * 100, bins=20, alpha=0.7, color='orange', label='NSGA-III')
    plt.xlabel('Penalty (%)')
    plt.ylabel('Frequency')
    plt.title('Contact Pressure Penalty Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Contact pressure distribution
    plt.subplot(2, 2, 2)
    if not valid_bo.empty:
        plt.hist(valid_bo['max_contact_pressure'], bins=20, alpha=0.7, color='blue', label='BO')
    if not valid_nsga3.empty:
        plt.hist(valid_nsga3['max_contact_pressure'], bins=20, alpha=0.7, color='orange', label='NSGA-III')
    plt.axvline(x=pressure_threshold_mpa, color='red', linestyle='--',
                label=f'Threshold ({pressure_threshold_mpa} MPa)')
    plt.xlabel('Max Contact Pressure (MPa)')
    plt.ylabel('Frequency')
    plt.title('Max Contact Pressure Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Total vs Penalized loss
    plt.subplot(2, 1, 2)
    if not df_bo.empty:
        plt.scatter(df_bo['total'], df_bo['penalized_total'], color='blue', alpha=0.6, s=30, label='BO')
    if not df_nsga3.empty:
        plt.scatter(df_nsga3['total'], df_nsga3['penalized_total'], color='orange', alpha=0.6, s=30, label='NSGA-III')

    # Add diagonal line
    if not df_bo.empty or not df_nsga3.empty:
        all_data = pd.concat([df_bo, df_nsga3]) if not df_bo.empty and not df_nsga3.empty else (
            df_bo if not df_bo.empty else df_nsga3)
        min_val = all_data['total'].min()
        max_val = all_data['total'].max()
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No Penalty Line')

    plt.xlabel('Total Loss [W]')
    plt.ylabel('Penalized Total Loss [W]')
    plt.title('Total Loss vs Penalized Total Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "penalty_analysis_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Penalty distribution analysis saved")


def create_plot_comparison_param_evolution_bo_nsga(df_bo, df_nsga3, output_dir):
    """Create parameter evolution plot comparing BO and NSGA-III with all points and best markers."""

    # Ensure consistent iteration column
    if 'generation' in df_nsga3.columns:
        df_nsga3 = df_nsga3.rename(columns={'generation': 'iteration'})

    # Process dataframes
    processed_dfs = []
    optimizer_names = []

    for df, name in [(df_bo, "BO"), (df_nsga3, "NSGA-III")]:
        if not df.empty:
            df_copy = df.copy()
            if 'clearance' not in df_copy.columns and 'dK' in df_copy.columns and 'dZ' in df_copy.columns:
                df_copy['clearance'] = df_copy['dZ'] - df_copy['dK']
            processed_dfs.append(df_copy)
            optimizer_names.append(name)
        else:
            processed_dfs.append(pd.DataFrame())
            optimizer_names.append(name)

    # Find best overall solutions
    best_overall_list = []
    for df in processed_dfs:
        if not df.empty and 'total' in df.columns:
            best_overall_idx = df['total'].idxmin()
            best_overall_list.append(df.loc[best_overall_idx])
        else:
            best_overall_list.append(None)

    # Parameters to plot
    parameters_to_plot = ["clearance", "LKG", "lF", "zeta"]
    colors = ['blue', 'orange']  # BO blue, NSGA-III orange
    markers = ['o', '^']  # BO circle, NSGA-III triangle

    if any(not df.empty for df in processed_dfs):
        fig, axes = plt.subplots(len(parameters_to_plot), 1, figsize=(12, 16), sharex=True)
        if len(parameters_to_plot) == 1:
            axes = [axes]

        handle_map = {}  # For legend

        for i, param in enumerate(parameters_to_plot):
            for j, (df, opt_name) in enumerate(zip(processed_dfs, optimizer_names)):
                if not df.empty and param in df.columns and 'iteration' in df.columns and 'total' in df.columns:
                    try:
                        # Add jitter to reduce overlap
                        jitter = np.random.normal(0, 0.15, len(df))

                        # All points
                        sc = axes[i].scatter(
                            df['iteration'] + jitter, df[param],
                            alpha=0.6, s=20, color=colors[j],
                            label=f'{opt_name} All Points',
                            edgecolors='none', marker=markers[j]
                        )

                        # Best per iteration/generation
                        best_indices = df.groupby('iteration')['total'].idxmin()
                        df_best = df.loc[best_indices]
                        ln, = axes[i].plot(
                            df_best['iteration'], df_best[param],
                            marker=markers[j], linestyle='-', color=colors[j],
                            label=f'{opt_name} Best per Iteration',
                            linewidth=2, markersize=8,
                            markeredgecolor='white', markeredgewidth=1
                        )

                        # Best overall
                        star = None
                        if best_overall_list[j] is not None and param in best_overall_list[j]:
                            star_color = 'red' if j == 0 else 'darkred'
                            star = axes[i].scatter(
                                [best_overall_list[j]['iteration']],
                                [best_overall_list[j][param]],
                                s=200, color=star_color,
                                marker='*', edgecolors='black', linewidth=2,
                                label=f'Best {opt_name} Overall', zorder=10
                            )

                        # Collect legend entries only once
                        if i == 0:
                            for h in [sc, ln] + ([star] if star is not None else []):
                                if h is not None:
                                    lab = h.get_label()
                                    if lab not in handle_map:
                                        handle_map[lab] = h
                    except Exception as e:
                        print(f"Error plotting {opt_name} {param}: {e}")
                        continue

            # Y-axis labels
            if param == "zeta":
                y_label = "γ"
            elif param == "clearance":
                y_label = "clearance [μm]"
            elif param == "LKG":
                y_label = "LKG [mm]"
            elif param == "lF":
                y_label = "lF [mm]"
            else:
                y_label = param

            axes[i].set_ylabel(y_label, fontsize=18)
            axes[i].tick_params(axis='both', labelsize=18)
            axes[i].grid(True, alpha=0.3)

        # X label only on the last subplot
        axes[-1].set_xlabel("Optimization Step (Iteration/Generation)", fontsize=18)
        axes[-1].tick_params(axis='both', labelsize=18)

        # Add combined legend below all plots
        handles = list(handle_map.values())
        labels = list(handle_map.keys())
        fig.legend(handles, labels, fontsize=14, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.02))

        # Title and layout
        fig.suptitle("Parameter Evolution: BO vs NSGA-III - All Points vs Best Points",
                     fontsize=16, y=0.995)
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        fig.savefig(output_dir / "comparison_param_evolution_bo_nsga.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Created comparison parameter evolution plot")
        return True

    return False


def create_best_solutions_comparison_bar_plot(df_bo, df_nsga3, output_dir, include_penalty=True):
    """Create a bar plot comparing the best solutions from BO and NSGA-III."""

    best_solutions = []
    technique_names = []

    # Process BO results
    if not df_bo.empty and 'total' in df_bo.columns:
        valid_bo = df_bo[(df_bo['total'] > 0) & (df_bo['total'] < 1e6)]
        if not valid_bo.empty:
            best_bo_total = valid_bo['total'].min()
            best_solutions.append(best_bo_total)
            technique_names.append('BO')

            if include_penalty and 'penalized_total' in valid_bo.columns:
                best_bo_penalized = valid_bo['penalized_total'].min()
                best_solutions.append(best_bo_penalized)
                technique_names.append('BO (w/ penalty)')

    # Process NSGA-III results
    if not df_nsga3.empty and 'total' in df_nsga3.columns:
        valid_nsga3 = df_nsga3[(df_nsga3['total'] > 0) & (df_nsga3['total'] < 1e6)]
        if not valid_nsga3.empty:
            best_nsga3_total = valid_nsga3['total'].min()
            best_solutions.append(best_nsga3_total)
            technique_names.append('NSGA-III')

            if include_penalty and 'penalized_total' in valid_nsga3.columns:
                best_nsga3_penalized = valid_nsga3['penalized_total'].min()
                best_solutions.append(best_nsga3_penalized)
                technique_names.append('NSGA-III (w/ penalty)')

    if best_solutions:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Define colors
        colors = []
        for name in technique_names:
            if 'BO' in name and 'penalty' in name:
                colors.append('darkblue')
            elif 'BO' in name:
                colors.append('blue')
            elif 'NSGA-III' in name and 'penalty' in name:
                colors.append('darkorange')
            else:
                colors.append('orange')

        bars = ax.bar(technique_names, best_solutions,
                      color=colors,
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels on top of bars
        for bar, value in zip(bars, best_solutions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.2e}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_xlabel('Optimization Method', fontsize=18)
        ax.set_ylabel('Best Total Loss [W]', fontsize=18)
        ax.set_title('Best Solution Comparison: BO vs NSGA-III', fontsize=16)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x labels if needed
        if len(technique_names) > 2:
            plt.xticks(rotation=45, ha='right', fontsize=16)
        else:
            plt.xticks(fontsize=18)

        plt.yticks(fontsize=18)
        plt.tight_layout()

        # Save the plot
        filename = "best_solutions_comparison_bar_plot.png" if include_penalty else "best_solutions_comparison_bar_plot_no_penalty.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created best solutions comparison bar plot")
        return True

    return False

# --- Main Orchestration and Utility Functions ---

def create_all_plots_parallel(df_bo, df_nsga3, output_dir, pressure_threshold_mpa=100.0, max_workers=None):
    """Create all plots in parallel using ThreadPoolExecutor for I/O bound plotting."""
    parameters = ["dK", "dZ", "LKG", "lF", "zeta"]

    # Ensure a common iteration column name for both
    if 'generation' in df_nsga3.columns:
        df_nsga3 = df_nsga3.rename(columns={'generation': 'iteration'})

    # Define all plotting tasks
    plot_functions = [
        (create_plot_convergence_bo, (df_bo, output_dir)),
        (create_plot_convergence_nsga3, (df_nsga3, output_dir)),
        (create_plot_param_distribution_bo, (df_bo, output_dir, parameters)),
        (create_plot_param_distribution_nsga3, (df_nsga3, output_dir, parameters)),
        (create_plot_param_vs_loss_bo, (df_bo, output_dir, parameters)),
        (create_plot_param_vs_loss_nsga3, (df_nsga3, output_dir, parameters)),
        (create_plot_pairplot_bo, (df_bo, output_dir, parameters)),
        (create_plot_pairplot_nsga3, (df_nsga3, output_dir, parameters)),
        (create_plot_pareto_bo, (df_bo, output_dir)),
        (create_plot_pareto_nsga3, (df_nsga3, output_dir)),
        (create_plot_combined_param_evolution_with_clearance, (df_bo, df_nsga3, output_dir)),
        (create_plot_all_points_param_evolution_with_clearance, (df_bo, df_nsga3, output_dir)),
        (create_plot_combined_param_evolution_original, (df_bo, df_nsga3, output_dir)),
        (create_plot_combined_convergence, (df_bo, df_nsga3, output_dir)),
        (create_plot_combined_pareto, (df_bo, df_nsga3, output_dir)),
        (create_plot_contact_pressure_vs_loss_analysis, (df_bo, df_nsga3, output_dir, pressure_threshold_mpa)),

        (create_plot_comparison_param_evolution_bo_nsga, (df_bo, df_nsga3, output_dir)),
        (create_best_solutions_comparison_bar_plot, (df_bo, df_nsga3, output_dir, True)),
        (create_best_solutions_comparison_bar_plot, (df_bo, df_nsga3, output_dir, False))

    ]

    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(plot_functions))

    print(f"Creating {len(plot_functions)} plots in parallel with {max_workers} workers...")

    # Use ThreadPoolExecutor for plotting since matplotlib/seaborn operations are mostly I/O bound
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all plotting tasks
        future_to_plot = {executor.submit(func, args): func.__name__
                          for func, args in plot_functions}

        # Collect results
        completed_plots = []
        for future in as_completed(future_to_plot):
            plot_name = future_to_plot[future]
            try:
                filename, status = future.result()
                completed_plots.append((plot_name, filename, status))
                print(f"  {plot_name}: {status}")
            except Exception as e:
                print(f"  {plot_name}: Failed with error {e}")

    print(f"Parallel plotting completed: {len(completed_plots)} plots generated")
    return completed_plots


def print_detailed_best_solutions(df_bo, df_nsga3):
    """Print detailed comparison of best solutions with and without penalty."""
    print(f"\nBEST SOLUTIONS COMPARISON (WITH vs WITHOUT PENALTY):")
    print("=" * 80)

    if not df_bo.empty:
        # Best with penalty
        best_bo_penalty_idx = df_bo['penalized_total'].idxmin()
        best_bo_penalty = df_bo.loc[best_bo_penalty_idx]

        # Best without penalty
        best_bo_no_penalty_idx = df_bo['total'].idxmin()
        best_bo_no_penalty = df_bo.loc[best_bo_no_penalty_idx]

        print(f"\nBO BEST SOLUTIONS:")
        print(f"   WITH PENALTY (Iteration {best_bo_penalty.get('iteration', 'N/A')}):")
        print(f"      - Total Loss: {best_bo_penalty['total']:.4e} W")
        print(f"      - Penalized Total: {best_bo_penalty['penalized_total']:.4e} W")
        print(f"      - Max Contact Pressure: {best_bo_penalty['max_contact_pressure']:.1f} MPa")
        print(f"      - Penalty: {best_bo_penalty['contact_pressure_penalty'] * 100:.1f}%")
        print(
            f"      - Parameters: dK={best_bo_penalty.get('dK', 0):.3f}, dZ={best_bo_penalty.get('dZ', 0):.3f}, clearance={best_bo_penalty.get('clearance', 0):.3f}")

        print(f"   WITHOUT PENALTY (Iteration {best_bo_no_penalty.get('iteration', 'N/A')}):")
        print(f"      - Total Loss: {best_bo_no_penalty['total']:.4e} W")
        print(f"      - Penalized Total: {best_bo_no_penalty['penalized_total']:.4e} W")
        print(f"      - Max Contact Pressure: {best_bo_no_penalty['max_contact_pressure']:.1f} MPa")
        print(f"      - Penalty: {best_bo_no_penalty['contact_pressure_penalty'] * 100:.1f}%")
        print(
            f"      - Parameters: dK={best_bo_no_penalty.get('dK', 0):.3f}, dZ={best_bo_no_penalty.get('dZ', 0):.3f}, clearance={best_bo_no_penalty.get('clearance', 0):.3f}")

    if not df_nsga3.empty:
        # Best with penalty
        best_nsga3_penalty_idx = df_nsga3['penalized_total'].idxmin()
        best_nsga3_penalty = df_nsga3.loc[best_nsga3_penalty_idx]

        # Best without penalty
        best_nsga3_no_penalty_idx = df_nsga3['total'].idxmin()
        best_nsga3_no_penalty = df_nsga3.loc[best_nsga3_no_penalty_idx]

        print(f"\nNSGA-III BEST SOLUTIONS:")
        print(f"   WITH PENALTY (Generation {best_nsga3_penalty.get('iteration', 'N/A')}):")
        print(f"      - Total Loss: {best_nsga3_penalty['total']:.4e} W")
        print(f"      - Penalized Total: {best_nsga3_penalty['penalized_total']:.4e} W")
        print(f"      - Max Contact Pressure: {best_nsga3_penalty['max_contact_pressure']:.1f} MPa")
        print(f"      - Penalty: {best_nsga3_penalty['contact_pressure_penalty'] * 100:.1f}%")
        print(
            f"      - Parameters: dK={best_nsga3_penalty.get('dK', 0):.3f}, dZ={best_nsga3_penalty.get('dZ', 0):.3f}, clearance={best_nsga3_penalty.get('clearance', 0):.3f}")

        print(f"   WITHOUT PENALTY (Generation {best_nsga3_no_penalty.get('iteration', 'N/A')}):")
        print(f"      - Total Loss: {best_nsga3_no_penalty['total']:.4e} W")
        print(f"      - Penalized Total: {best_nsga3_no_penalty['penalized_total']:.4e} W")
        print(f"      - Max Contact Pressure: {best_nsga3_no_penalty['max_contact_pressure']:.1f} MPa")
        print(f"      - Penalty: {best_nsga3_no_penalty['contact_pressure_penalty'] * 100:.1f}%")
        print(
            f"      - Parameters: dK={best_nsga3_no_penalty.get('dK', 0):.3f}, dZ={best_nsga3_no_penalty.get('dZ', 0):.3f}, clearance={best_nsga3_no_penalty.get('clearance', 0):.3f}")


def print_top_results_summary(df_bo, df_nsga3, top_n_display=10):
    """Display top optimization results summary similar to original code."""
    print("\n--- Top 10 Optimal Results from Pareto Front (NSGA-III) and Lowest Total Loss (BO) ---")

    columns_to_show = ["optimizer", "total", "mechanical", "volumetric", "dK", "dZ", "LKG", "lF", "zeta", "folder_name"]

    # For BO: Get top N by total loss
    if not df_bo.empty:
        top_bo_results = df_bo.nsmallest(top_n_display, 'total').copy()
        top_bo_results["Optimizer Selection"] = "Lowest Total Loss"
        top_bo_results = top_bo_results[columns_to_show]
        print(f"\nBO Top {len(top_bo_results)} (by lowest total loss):")
        if not top_bo_results.empty:
            for i, row in top_bo_results.iterrows():
                print(
                    f"{i + 1}. Total Loss: {row['total']:.4e}, Mech Loss: {row['mechanical']:.4e}, Vol Loss: {row['volumetric']:.4e}")
                print(
                    f"    Parameters: dK={row['dK']:.4f}, dZ={row['dZ']:.4f}, LKG={row['LKG']:.4f}, lF={row['lF']:.4f}, zeta={row['zeta']:.0f}")
        else:
            print("  No valid BO results to display.")

    # For NSGA-III: Find Pareto front and then select top N by total loss from Pareto front
    if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
        pareto_nsga3 = find_pareto_front(df_nsga3, ["mechanical", "volumetric"])

        if not pareto_nsga3.empty:
            top_nsga3_results = pareto_nsga3.sort_values(by='total').head(top_n_display).copy()
            top_nsga3_results["Optimizer Selection"] = "Pareto Front (Sorted by Total Loss)"
            top_nsga3_results = top_nsga3_results[columns_to_show]
            print(f"\nNSGA-III Top {top_nsga3_results.shape[0]} (from Pareto Front, sorted by total loss):")
            if not top_nsga3_results.empty:
                for i, row in top_nsga3_results.iterrows():
                    print(
                        f"{i + 1}. Total Loss: {row['total']:.4e}, Mech Loss: {row['mechanical']:.4e}, Vol Loss: {row['volumetric']:.4e}")
                    print(
                        f"    Parameters: dK={row['dK']:.4f}, dZ={row['dZ']:.4f}, LKG={row['LKG']:.4f}, lF={row['lF']:.4f}, zeta={row['zeta']:.0f}")
        else:
            print("  No Pareto optimal solutions found for NSGA-III.")
    else:
        print("  NSGA-III data is empty or missing 'mechanical'/'volumetric' columns.")


def save_optimization_results_parallel(df_bo, df_nsga3, output_dir, top_n_save=5):
    """Save optimization results in parallel."""
    columns_to_save = ["optimizer", "iteration", "total", "penalized_total", "mechanical", "volumetric",
                       "max_contact_pressure", "avg_contact_pressure", "contact_pressure_penalty",
                       "dK", "dZ", "clearance", "LKG", "lF", "zeta", "pattern_type", "contact_pressure_valid"]

    columns_to_save_traditional = ["optimizer", "iteration", "total", "mechanical", "volumetric", "dK", "dZ", "LKG",
                                   "lF", "zeta"]

    def prepare_top_results(df, optimizer_name, top_n, penalty_col='penalized_total', loss_col='total'):
        """Prepare top results for a single optimizer."""
        results = []
        if not df.empty:
            # Add clearance calculation
            df_copy = df.copy()
            if 'clearance' not in df_copy.columns:
                df_copy['clearance'] = df_copy['dZ'] - df_copy['dK']

            # Top results with penalty
            top_penalty = df_copy.nsmallest(top_n, penalty_col).copy()
            top_penalty["optimizer"] = optimizer_name
            results.append(top_penalty[columns_to_save])

            # Top results without penalty
            top_no_penalty = df_copy.nsmallest(top_n, loss_col).copy()
            top_no_penalty["optimizer"] = f"{optimizer_name}_no_penalty"
            results.append(top_no_penalty[columns_to_save])

            # Traditional format
            top_traditional = df_copy.nsmallest(top_n, loss_col).copy()
            top_traditional["optimizer"] = optimizer_name
            results.append(top_traditional[columns_to_save_traditional])

        return results

    # Prepare results in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bo = executor.submit(prepare_top_results, df_bo, "BO", top_n_save)
        future_nsga3 = executor.submit(prepare_top_results, df_nsga3, "NSGA-III", top_n_save)

        bo_results = future_bo.result()
        nsga3_results = future_nsga3.result()

    # Combine and save results
    all_results = bo_results + nsga3_results

    if all_results:
        # Save enhanced results (with penalty)
        enhanced_results = [result for i, result in enumerate(all_results) if i % 3 != 2]  # Skip traditional results
        if enhanced_results:
            df_enhanced = pd.concat(enhanced_results, ignore_index=True)
            df_enhanced.to_csv(output_dir / "top_optimal_results_with_contact_penalty.csv", index=False)

        # Save traditional results
        traditional_results = [result for i, result in enumerate(all_results) if i % 3 == 2]  # Only traditional results
        if traditional_results:
            df_traditional = pd.concat(traditional_results, ignore_index=True)
            df_traditional.to_csv(output_dir / "top5_optimal_results.csv", index=False)



def add_comparison_plots_to_parallel_execution(df_bo, df_nsga3, output_dir):
    """Add the comparison plots to the parallel plotting execution."""
    # Add these function calls to your create_all_plots_parallel function
    plot_functions = [
        (create_plot_comparison_param_evolution_bo_nsga, (df_bo, df_nsga3, output_dir)),
        (create_best_solutions_comparison_bar_plot, (df_bo, df_nsga3, output_dir, True)),
        (create_best_solutions_comparison_bar_plot, (df_bo, df_nsga3, output_dir, False))
    ]
    return plot_functions


def print_best_parameters_all_phases(df_list, folder_names):
    """Print the best parameters for each phase."""

    # Create phase labels based on folder order
    phase_labels = []
    for i, folder_name in enumerate(folder_names):
        if i == 0:
            phase_labels.append("Phase I")
        elif i == 1:
            phase_labels.append("Phase II")
        elif i == 2:
            phase_labels.append("Phase III")
        else:
            phase_labels.append(f"Phase {i + 1}")

    print("\n" + "=" * 80)
    print("BEST PARAMETERS FOR ALL PHASES")
    print("=" * 80)

    # Parameters to display
    param_columns = ['dK', 'dZ', 'clearance', 'LKG', 'lF', 'zeta', 'total', 'mechanical', 'volumetric',
                     'max_contact_pressure', 'contact_pressure_penalty', 'penalized_total', 'iteration']

    for df, phase_label in zip(df_list, phase_labels):
        if not df.empty and 'total' in df.columns:
            # Filter valid data
            valid_data = df[(df['total'] > 0) & (df['total'] < 1e6)]

            if not valid_data.empty:
                # Find best solution (minimum total loss)
                best_idx = valid_data['total'].idxmin()
                best_solution = valid_data.loc[best_idx]

                print(f"\n{phase_label} BEST SOLUTION:")
                print("-" * 40)
                print(f"Generation/Iteration: {best_solution.get('iteration', 'N/A')}")
                print(f"Total Loss: {best_solution['total']:.6e} W")
                print(f"Mechanical Loss: {best_solution.get('mechanical', 'N/A'):.6e} W")
                print(f"Volumetric Loss: {best_solution.get('volumetric', 'N/A'):.6e} W")

                if 'penalized_total' in best_solution and pd.notna(best_solution['penalized_total']):
                    print(f"Penalized Total: {best_solution['penalized_total']:.6e} W")

                print(f"\nGeometry Parameters:")
                if 'dK' in best_solution and pd.notna(best_solution['dK']):
                    print(f"  dK (Piston Diameter): {best_solution['dK']:.6f} mm")
                if 'dZ' in best_solution and pd.notna(best_solution['dZ']):
                    print(f"  dZ (Bushing Diameter): {best_solution['dZ']:.6f} mm")
                if 'clearance' in best_solution and pd.notna(best_solution['clearance']):
                    print(f"  Clearance: {best_solution['clearance']:.6f} mm")

                print(f"\nOther Parameters:")
                if 'LKG' in best_solution and pd.notna(best_solution['LKG']):
                    print(f"  LKG (Leak Gap): {best_solution['LKG']:.6f} mm")
                if 'lF' in best_solution and pd.notna(best_solution['lF']):
                    print(f"  lF (Length): {best_solution['lF']:.6f} mm")
                if 'zeta' in best_solution and pd.notna(best_solution['zeta']):
                    print(f"  zeta (γ): {best_solution['zeta']:.0f}")

                print(f"\nContact Pressure Analysis:")
                if 'max_contact_pressure' in best_solution and pd.notna(best_solution['max_contact_pressure']):
                    print(f"  Max Contact Pressure: {best_solution['max_contact_pressure']:.3f} MPa")
                if 'contact_pressure_penalty' in best_solution and pd.notna(best_solution['contact_pressure_penalty']):
                    print(f"  Contact Pressure Penalty: {best_solution['contact_pressure_penalty'] * 100:.2f}%")

                # Also find and print the best with penalty consideration
                if 'penalized_total' in valid_data.columns and valid_data['penalized_total'].notna().any():
                    best_penalty_idx = valid_data['penalized_total'].idxmin()
                    best_penalty_solution = valid_data.loc[best_penalty_idx]

                    if best_penalty_idx != best_idx:  # Only show if different from best total loss
                        print(f"\n{phase_label} BEST WITH PENALTY CONSIDERATION:")
                        print("-" * 40)
                        print(f"Generation/Iteration: {best_penalty_solution.get('iteration', 'N/A')}")
                        print(f"Total Loss: {best_penalty_solution['total']:.6e} W")
                        print(f"Penalized Total: {best_penalty_solution['penalized_total']:.6e} W")
                        print(f"Max Contact Pressure: {best_penalty_solution.get('max_contact_pressure', 0):.3f} MPa")
                        print(
                            f"Contact Pressure Penalty: {best_penalty_solution.get('contact_pressure_penalty', 0) * 100:.2f}%")

                        print(
                            f"Geometry: dK={best_penalty_solution.get('dK', 0):.6f}, dZ={best_penalty_solution.get('dZ', 0):.6f}, clearance={best_penalty_solution.get('clearance', 0):.6f}")
                        print(
                            f"Other: LKG={best_penalty_solution.get('LKG', 0):.6f}, lF={best_penalty_solution.get('lF', 0):.6f}, zeta={best_penalty_solution.get('zeta', 0):.0f}")
            else:
                print(f"\n{phase_label}: No valid solutions found")
        else:
            print(f"\n{phase_label}: No data available")

    print("\n" + "=" * 80)

# --- Main Function ---

def main_parallel():
    """Parallel version of the main function."""
    start_time = time.time()

    # Configuration
    bo_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\bayesian_optimization'
    # nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\advanced_nsga3'
    nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\simple_nsga3'

    pressure_threshold_mpa = 10.0
    max_penalty_percent = 200.0

    # Determine optimal number of workers
    num_workers = min(mp.cpu_count(), 24)  # Cap at 8 for memory management
    print(f"Using {num_workers} parallel workers (out of {mp.cpu_count()} available CPUs)")

    print("=" * 80)
    print("PARALLEL OPTIMIZATION ANALYSIS WITH CONTACT PRESSURE PENALTY")
    print("=" * 80)
    print("Supported formats:")
    print("   - NEW: CL0.028_dZ19.888_LKG56.4_lF36.9_zeta5 (extracts dK from geometry.txt)")
    print("   - OLD: dK19.48_dZ19.51_LKG56.4_lF36.9_zeta5 (calculates clearance as dZ-dK)")
    print(f"Contact Pressure Penalty Settings:")
    print(f"   - Pressure Threshold: {pressure_threshold_mpa:.1f} MPa")
    print(f"   - Maximum Penalty: {max_penalty_percent:.0f}%")
    print("=" * 80)

    # Run diagnostics first
    print("\nRUNNING DIAGNOSTICS...")
    diagnose_folder_structure(bo_folder, nsga3_folder)

    # Load results in parallel
    print("\nLOADING OPTIMIZATION RESULTS WITH CONTACT PRESSURE ANALYSIS")
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
    df_bo = pd.DataFrame(bo_results)
    df_nsga3 = pd.DataFrame(nsga3_results)

    # Ensure common iteration column
    if 'generation' in df_nsga3.columns:
        df_nsga3 = df_nsga3.rename(columns={'generation': 'iteration'})

    # Create output directory
    output_dir = Path("optimization_plots_parallel")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print data summary
    print(f"\nDATA SUMMARY WITH CONTACT PRESSURE ANALYSIS:")
    if not df_bo.empty:
        print(f"   BO data shape: {df_bo.shape}")
        pattern_counts_bo = df_bo['pattern_type'].value_counts()
        for pattern, count in pattern_counts_bo.items():
            print(f"     - {pattern} format: {count} simulations")

        valid_pressure_bo = df_bo[df_bo['contact_pressure_valid'] == True]
        if not valid_pressure_bo.empty:
            print(f"   BO Contact Pressure Statistics:")
            print(f"     - Valid contact pressure data: {len(valid_pressure_bo)}/{len(df_bo)} simulations")
            print(
                f"     - Max contact pressure range: {valid_pressure_bo['max_contact_pressure'].min():.1f} - {valid_pressure_bo['max_contact_pressure'].max():.1f} MPa")
            print(f"     - Average penalty: {valid_pressure_bo['contact_pressure_penalty'].mean() * 100:.1f}%")

    if not df_nsga3.empty:
        print(f"   NSGA-III data shape: {df_nsga3.shape}")
        pattern_counts_nsga3 = df_nsga3['pattern_type'].value_counts()
        for pattern, count in pattern_counts_nsga3.items():
            print(f"     - {pattern} format: {count} simulations")

        valid_pressure_nsga3 = df_nsga3[df_nsga3['contact_pressure_valid'] == True]
        if not valid_pressure_nsga3.empty:
            print(f"   NSGA-III Contact Pressure Statistics:")
            print(f"     - Valid contact pressure data: {len(valid_pressure_nsga3)}/{len(df_nsga3)} simulations")
            print(
                f"     - Max contact pressure range: {valid_pressure_nsga3['max_contact_pressure'].min():.1f} - {valid_pressure_nsga3['max_contact_pressure'].max():.1f} MPa")
            print(f"     - Average penalty: {valid_pressure_nsga3['contact_pressure_penalty'].mean() * 100:.1f}%")

    # Create all plots in parallel
    print("\nCREATING ENHANCED PLOTS WITH CONTACT PRESSURE PENALTY ANALYSIS")
    print("=" * 60)
    plot_start_time = time.time()

    # Create different plot groups in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit different plotting tasks
        future_basic_plots = executor.submit(create_all_plots_parallel, df_bo, df_nsga3, output_dir,
                                             pressure_threshold_mpa, num_workers // 2)
        future_enhanced_plots = executor.submit(create_enhanced_combined_plots_with_penalty_parallel,
                                                df_bo, df_nsga3, output_dir, pressure_threshold_mpa)
        future_sensitivity = executor.submit(create_sensitivity_analysis_parallel, df_bo, df_nsga3, output_dir)
        future_penalty_analysis = executor.submit(create_penalty_distribution_analysis_parallel,
                                                  df_bo, df_nsga3, output_dir, pressure_threshold_mpa)

        # Wait for completion
        basic_plots_result = future_basic_plots.result()
        future_enhanced_plots.result()
        future_sensitivity.result()
        future_penalty_analysis.result()

    plot_time = time.time() - plot_start_time
    print(f"Parallel plotting completed in {plot_time:.2f} seconds")

    # Save results in parallel
    print("\nSaving optimization results...")
    save_start_time = time.time()
    save_optimization_results_parallel(df_bo, df_nsga3, output_dir, top_n_save=5)
    save_time = time.time() - save_start_time

    # Print detailed best solution comparison
    print_detailed_best_solutions(df_bo, df_nsga3)

    # Display top results summary
    print_top_results_summary(df_bo, df_nsga3, top_n_display=10)

    total_time = time.time() - start_time

    # Print performance summary
    print("\n" + "=" * 80)
    print("PARALLEL OPTIMIZATION ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"Performance Summary:")
    print(f"  Data loading time: {load_time:.2f} seconds")
    print(f"  Plotting time: {plot_time:.2f} seconds")
    print(f"  Results saving time: {save_time:.2f} seconds")
    print(f"  Total execution time: {total_time:.2f} seconds")
    print(f"  Speedup achieved through parallelization")
    print(f"All results saved to: {output_dir}")
    print(f"Generated plots:")
    print(f"   - Enhanced convergence plots with penalty comparison")
    print(f"   - Contact pressure vs loss analysis")
    print(f"   - Penalty distribution analysis")
    print(f"   - Individual convergence plots (BO and NSGA-III)")
    print(f"   - Combined parameter evolution with clearance and convergence")
    print(f"   - All points parameter evolution (showing all selected points)")
    print(f"   - Parameter distribution histograms")
    print(f"   - Parameter vs total loss scatter plots")
    print(f"   - Pairwise parameter relationship plots")
    print(f"   - Pareto front plots")
    print(f"   - Combined plots (convergence, pareto, parameter evolution)")
    print(f"   - Sensitivity analysis bar chart")
    print("Generated files:")
    print(f"   - top_optimal_results_with_contact_penalty.csv")
    print(f"   - top5_optimal_results.csv")
    print("=" * 80)
    print("Complete optimization analysis with contact pressure penalty and all plots completed!")



if __name__ == "__main__":
    # Set start method for multiprocessing (important for Windows)
    mp.set_start_method('spawn', force=True)
    main_parallel()