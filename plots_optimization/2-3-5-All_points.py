import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
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
import os
import time

# Import helper functions from the neighbouring T1 module while keeping
# compatibility with running this script directly.
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

PARAM_PATTERN_GENERATION = re.compile(
    r"T_(\d+)_(\d+)_C(\d+)of(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_dK_(\d+(?:\.\d+)?)_dZ_(\d+(?:\.\d+)?)"
)


def detect_folder_pattern(folder_name: str) -> Tuple[str, Optional[re.Match]]:
    """Detect which pattern the folder name matches and return pattern type and match object."""
    match_new = PARAM_PATTERN_NEW.search(folder_name)
    if match_new:
        return "new", match_new

    match_old = PARAM_PATTERN_OLD.search(folder_name)
    if match_old:
        return "old", match_old

    return "none", None


def detect_folder_pattern_extended(folder_name: str) -> Tuple[str, Optional[re.Match]]:
    """Extended version that detects which pattern the folder name matches including generation pattern."""
    match_new = PARAM_PATTERN_NEW.search(folder_name)
    if match_new:
        return "new", match_new

    match_old = PARAM_PATTERN_OLD.search(folder_name)
    if match_old:
        return "old", match_old

    match_generation = PARAM_PATTERN_GENERATION.search(folder_name)
    if match_generation:
        return "generation", match_generation

    return "none", None


def extract_geometry_parameters_generation(sim_folder_path: str) -> Dict[str, float]:
    """Extract geometry parameters from input/geometry.txt for generation pattern folders."""
    geometry_file = Path(sim_folder_path) / "input" / "geometry.txt"

    if not geometry_file.exists():
        return {}

    try:
        with open(geometry_file, 'r') as file:
            content = file.read()

        params = {}

        # Look for LKG (leak gap)
        lkg_pattern = re.compile(r'LKG\s+([+-]?(?:\d+\.?\d*|\.\d+))')
        lkg_match = lkg_pattern.search(content)
        if lkg_match:
            params["LKG"] = float(lkg_match.group(1))

        # Look for lF (length parameter)
        lf_pattern = re.compile(r'lF\s+([+-]?(?:\d+\.?\d*|\.\d+))')
        lf_match = lf_pattern.search(content)
        if lf_match:
            params["lF"] = float(lf_match.group(1))

        # Look for zeta (gamma parameter)
        zeta_pattern = re.compile(r'zeta\s+([+-]?(?:\d+\.?\d*|\.\d+))')
        zeta_match = zeta_pattern.search(content)
        if zeta_match:
            params["zeta"] = float(zeta_match.group(1))

        return params

    except Exception as e:
        return {}


def parse_geometry_file(geometry_file_path: str) -> Dict[str, Optional[float]]:
    """Extract dK and lF values from geometry.txt file."""
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
        return result


def load_contact_pressure(sim_path: str) -> Dict[str, float]:
    """Load contact pressure data from Piston_Contact_Pressure.txt file"""
    contact_pressure_path = Path(sim_path) / "output" / "piston" / "matlab" / "Piston_Contact_Pressure.txt"

    if not contact_pressure_path.exists():
        return {"max_contact_pressure": 0.0, "avg_contact_pressure": 0.0, "contact_pressure_valid": False}

    try:
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
        return {"max_contact_pressure": 0.0, "avg_contact_pressure": 0.0, "contact_pressure_valid": False}


def calculate_contact_pressure_penalty(max_contact_pressure_mpa: float,
                                       pressure_threshold_mpa: float = 100.0,
                                       max_penalty_percent: float = 200.0) -> float:
    """Calculate contact pressure penalty based on maximum contact pressure."""
    if max_contact_pressure_mpa <= 0:
        return 0.0

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
        return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}


def extract_data_from_folders(base_path):
    """Extract data from the nested folder structure and create a CSV file."""
    data = []
    base_path = Path(base_path)

    if not base_path.exists():
        return pd.DataFrame()

    # Get all Generation folders and sort them by generation number
    generation_folders = []
    for folder in base_path.glob("Generation_*"):
        if folder.is_dir():
            generation_match = re.search(r'Generation_(\d+)', folder.name)
            if generation_match:
                gen_num = int(generation_match.group(1))
                generation_folders.append((gen_num, folder))

    # Sort by generation number to process in order
    generation_folders.sort(key=lambda x: x[0])

    # Iterate through Generation folders in sorted order
    for generation, generation_folder in generation_folders:

        # Iterate through Individual folders
        individual_folders_found = list(generation_folder.glob("Individual_*"))

        for individual_folder in individual_folders_found:
            if not individual_folder.is_dir():
                continue

            # Extract individual info: Individual_{ind_value}_P_{phase_number}_{validation_type}_{number}C
            individual_match = re.search(r'Individual_(\d+)_P(\d+)_(Exp|Ref|Val)_(\d+)C', individual_folder.name)
            if not individual_match:
                continue

            individual = int(individual_match.group(1))
            phase = int(individual_match.group(2))
            validation_type = individual_match.group(3)
            condition_suffix = int(individual_match.group(4))

            # Map validation type to number
            validation_mapping = {'Exp': 1, 'Ref': 2, 'Val': 3}
            validation_num = validation_mapping.get(validation_type, 0)

            # Iterate through T_ folders (condition folders)
            condition_folders_found = list(individual_folder.glob("T_*"))

            for condition_folder in condition_folders_found:
                if not condition_folder.is_dir():
                    continue

                # Parse T_ folder name
                t_pattern = r'T_(\d+)_(\d+)_C(\d+)of(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_dK_(\d+(?:\.\d+)?)_dZ_(\d+(?:\.\d+)?)'
                t_match = re.search(t_pattern, condition_folder.name)

                if not t_match:
                    continue

                # Extract values
                gen_no = int(t_match.group(1))
                ind_no = int(t_match.group(2))
                condition_num = int(t_match.group(3))
                total_conditions = int(t_match.group(4))
                speed = float(t_match.group(5))
                pressure = float(t_match.group(6))
                displacement = float(t_match.group(7))
                dk_value = float(t_match.group(8))
                dz_value = float(t_match.group(9))

                # Create data entry
                data_entry = {
                    'generation': generation,
                    'individual': individual,
                    'condition_set': condition_num,
                    'total_sets': total_conditions,
                    'speed': speed,
                    'pressure': pressure,
                    'displacement': displacement,
                    'dk_value': dk_value,
                    'dz_value': dz_value,
                    'folder_path': str(condition_folder),
                    'phase': phase,
                    'validation_type': validation_type,
                    'validation_num': validation_num
                }

                data.append(data_entry)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by generation, individual, condition_set for better organization
    if not df.empty:
        df = df.sort_values(['generation', 'individual', 'phase', 'validation_num', 'condition_set'])
        df = df.reset_index(drop=True)

    return df


def process_single_simulation_generation_pattern(sim_folder_data):
    """Process a single simulation folder with generation pattern - designed for parallel execution."""
    condition_folder, generation, individual, phase, validation_type, validation_num, pressure_threshold_mpa, max_penalty_percent = sim_folder_data

    # Parse the T_ folder pattern
    t_pattern = r'T_(\d+)_(\d+)_C(\d+)of(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_dK_(\d+(?:\.\d+)?)_dZ_(\d+(?:\.\d+)?)'
    t_match = re.search(t_pattern, condition_folder.name)

    if not t_match:
        return None

    # Extract parameters from folder name
    gen_no = int(t_match.group(1))
    ind_no = int(t_match.group(2))
    condition_num = int(t_match.group(3))
    total_conditions = int(t_match.group(4))
    speed = float(t_match.group(5))
    pressure = float(t_match.group(6))
    displacement = float(t_match.group(7))
    dK = float(t_match.group(8))
    dZ = float(t_match.group(9))

    # Calculate additional parameters
    clearance = dZ - dK

    # Extract additional parameters from geometry.txt
    geometry_params = extract_geometry_parameters_generation(str(condition_folder))

    # Load losses
    losses = parse_loss_file(str(condition_folder))

    # Load contact pressure data
    contact_pressure_data = load_contact_pressure(str(condition_folder))

    # Calculate penalty
    penalty = calculate_contact_pressure_penalty(
        contact_pressure_data["max_contact_pressure"],
        pressure_threshold_mpa,
        max_penalty_percent
    )

    # Calculate penalized total loss
    penalized_total = losses["total"] * (1 + penalty)

    # Combine all data - ensuring iteration is a scalar value
    record = {
        'generation': int(generation),
        'individual': int(individual),
        'condition_set': int(condition_num),
        'total_sets': int(total_conditions),
        'speed': float(speed),
        'pressure': float(pressure),
        'displacement': float(displacement),
        'dK': float(dK),
        'dZ': float(dZ),
        'clearance': float(clearance),
        'phase': int(phase),
        'validation_type': str(validation_type),
        'validation_num': int(validation_num),
        'folder_path': str(condition_folder),
        'optimizer': 'Generation-NSGA-III',
        'pattern_type': 'generation',
        'iteration': int(generation),
        **losses,
        **contact_pressure_data,
        **geometry_params,
        'contact_pressure_penalty': float(penalty),
        'penalized_total': float(penalized_total)
    }

    if record.get("valid") and record["total"] < 1e6:
        return record
    return None


def load_results_generation_pattern_parallel(folder_path: str,
                                             pressure_threshold_mpa: float = 100.0,
                                             max_penalty_percent: float = 200.0,
                                             max_workers: int = None) -> list:
    """Load results from generation pattern folders in parallel."""
    df_structure = extract_data_from_folders(folder_path)

    if df_structure.empty:
        return []

    # Prepare simulation tasks
    simulation_tasks = []

    for _, row in df_structure.iterrows():
        condition_folder = Path(row['folder_path'])
        if condition_folder.exists():
            task = (
                condition_folder,
                row['generation'],
                row['individual'],
                row['phase'],
                row['validation_type'],
                row['validation_num'],
                pressure_threshold_mpa,
                max_penalty_percent
            )
            simulation_tasks.append(task)

    # Process simulations in parallel
    results = []
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(simulation_tasks))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_sim = {executor.submit(process_single_simulation_generation_pattern, task): task
                         for task in simulation_tasks}

        for future in as_completed(future_to_sim):
            result = future.result()
            if result is not None:
                results.append(result)

    return results


def parse_parameters_flexible(folder_name: str, folder_path: str) -> Dict[str, float]:
    """Flexible parameter parsing that handles both old and new folder naming conventions."""
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
                calculated_clearance = params["dZ"] - params["dK"]
                params["clearance"] = calculated_clearance
            else:
                params["dK"] = params["dZ"] - params["CL"]  # Fallback calculation
                params["clearance"] = params["CL"]

        elif pattern_type == "old":
            # OLD FORMAT: Folder has dK, calculate clearance as dZ - dK
            params["clearance"] = params["dZ"] - params["dK"]

            # If geometry file has dK, compare with folder value
            if geometry_params["dK"] is not None:
                geometry_dK = geometry_params["dK"]
                folder_dK = params["dK"]

                if abs(geometry_dK - folder_dK) > 0.001:
                    # Use geometry file value and recalculate clearance
                    params["dK"] = geometry_dK
                    params["clearance"] = params["dZ"] - params["dK"]

        # Handle lF parameter (always prefer geometry file if available)
        if geometry_params["lF"] is not None:
            params["lF"] = geometry_params["lF"]

        return params

    except ValueError as e:
        return {}


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

    # Combine all data - ensuring all values are scalars
    record = {**params, **losses, **contact_pressure_data}
    record["contact_pressure_penalty"] = float(penalty)
    record["penalized_total"] = float(penalized_total)
    record[iter_type] = int(iter_num)
    record["iteration"] = int(iter_num)
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
        return []

    subfolders = [f for f in base_folder.iterdir() if f.is_dir()]

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

    # Process simulations in parallel
    results = []
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(simulation_tasks))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_sim = {executor.submit(process_single_simulation, task): task
                         for task in simulation_tasks}

        for future in as_completed(future_to_sim):
            result = future.result()
            if result is not None:
                results.append(result)

    return results


def safe_groupby_operation(df, group_col, value_col, operation='min'):
    """Safely perform groupby operations with error handling."""
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    try:
        # Ensure the grouping column contains scalar values
        if not df[group_col].apply(np.isscalar).all():
            # Convert non-scalar values to scalars
            df[group_col] = df[group_col].apply(
                lambda x: x if np.isscalar(x) else (int(x) if hasattr(x, '__iter__') else 0))

        if operation == 'min':
            return df.groupby(group_col)[value_col].min().reset_index()
        elif operation == 'max':
            return df.groupby(group_col)[value_col].max().reset_index()
        elif operation == 'mean':
            return df.groupby(group_col)[value_col].mean().reset_index()
        else:
            return df.groupby(group_col)[value_col].agg(operation).reset_index()

    except Exception as e:
        return pd.DataFrame()


def prepare_dataframe_for_plotting(df, name):
    """Prepare a DataFrame for plotting by ensuring proper data types and structure."""
    if df.empty:
        return df

    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure iteration column is properly formatted
    if 'iteration' in df.columns:
        df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce').fillna(0).astype(int)

    # Ensure generation column is properly formatted if it exists
    if 'generation' in df.columns:
        df['generation'] = pd.to_numeric(df['generation'], errors='coerce').fillna(0).astype(int)
        # If iteration doesn't exist, use generation
        if 'iteration' not in df.columns:
            df['iteration'] = df['generation']

    # Ensure numeric columns are properly typed
    numeric_columns = ['dK', 'dZ', 'total', 'mechanical', 'volumetric',
                       'max_contact_pressure', 'contact_pressure_penalty', 'penalized_total']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add missing columns with defaults
    if 'clearance' not in df.columns and 'dK' in df.columns and 'dZ' in df.columns:
        df['clearance'] = df['dZ'] - df['dK']

    # Ensure required columns exist with defaults
    default_values = {
        'LKG': 0.0,
        'lF': 10.0,
        'zeta': 1.0,
        'contact_pressure_penalty': 0.0,
        'penalized_total': df['total'] if 'total' in df.columns else 1e6,
        'max_contact_pressure': 0.0,
        'contact_pressure_valid': False
    }

    for col, default_val in default_values.items():
        if col not in df.columns:
            if col == 'penalized_total' and 'total' in df.columns:
                df[col] = df['total']
            else:
                df[col] = default_val

    return df


def create_plot_all_points_param_evolution_with_clearance_three_nsga(df_list, folder_names, output_path):
    """Create parameter evolution plot showing ALL points for three NSGA-III runs with best markers.
    Produces a single combined legend centered below the x-axis of the last subplot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

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

    # Add clearance calculation for all dataframes
    processed_df_list = []
    for df in df_list:
        if df is not None and not df.empty:
            df = df.copy()
            if 'clearance' not in df.columns and 'dK' in df.columns and 'dZ' in df.columns:
                df['clearance'] = df['dZ'] - df['dK']
            processed_df_list.append(df)
        else:
            # keep placeholder for alignment with phases
            import pandas as pd
            processed_df_list.append(pd.DataFrame())

    # Find best overall solutions (across all iterations/generations)
    best_overall_list = []
    for df in processed_df_list:
        if not df.empty and 'total' in df.columns:
            best_overall_idx = df['total'].idxmin()
            best_overall_list.append(df.loc[best_overall_idx])
        else:
            best_overall_list.append(None)

    # Only show parameter plots (no convergence plot)
    parameters_to_plot = ["clearance", "LKG", "lF", "zeta"]
    colors = ['orange', 'green', 'purple']  # per phase
    markers = ['o', '^', 's']               # per phase

    # If any df has content, proceed
    if any(not df.empty for df in processed_df_list):
        fig, axes = plt.subplots(len(parameters_to_plot), 1, figsize=(12, 16), sharex=True)
        if len(parameters_to_plot) == 1:
            axes = [axes]

        # We'll gather legend handles/labels once (from first parameter only) to make a single legend
        handle_map = {}  # label -> handle; preserves uniqueness

        for i, param in enumerate(parameters_to_plot):
            for j, (df, phase_label) in enumerate(zip(processed_df_list, phase_labels)):
                if not df.empty and param in df.columns and 'iteration' in df.columns and 'total' in df.columns:
                    try:
                        # jitter to reduce overlap
                        jitter = np.random.normal(0, 0.15, len(df))
                        sc = axes[i].scatter(
                            df['iteration'] + jitter, df[param],
                            alpha=0.6, s=20, color=colors[j],
                            label=f'{phase_label} All Points',
                            edgecolors='none', marker=markers[j]
                        )

                        # Best per generation
                        best_indices = df.groupby('iteration')['total'].idxmin()
                        df_best = df.loc[best_indices]
                        ln, = axes[i].plot(
                            df_best['iteration'], df_best[param],
                            marker=markers[j], linestyle='-', color=colors[j],
                            label=f'{phase_label} Best per Generation',
                            linewidth=2, markersize=8,
                            markeredgecolor='white', markeredgewidth=1
                        )

                        # Best overall (if available)
                        star = None
                        if best_overall_list[j] is not None and param in best_overall_list[j]:
                            star = axes[i].scatter(
                                [best_overall_list[j]['iteration']],
                                [best_overall_list[j][param]],
                                s=200, color=['red', 'darkred', 'maroon'][j],
                                marker='*', edgecolors='black', linewidth=2,
                                label=f'Best {phase_label} Overall', zorder=10
                            )

                        # Collect legend entries only once (on first parameter)
                        if i == 0:
                            for h in [sc, ln] + ([star] if star is not None else []):
                                if h is not None:
                                    lab = h.get_label()
                                    if lab not in handle_map:
                                        handle_map[lab] = h
                    except Exception:
                        continue

            # Y-axis label per parameter
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
        axes[-1].set_xlabel("Generation", fontsize=18)
        axes[-1].tick_params(axis='both', labelsize=18)

        # Add ONE combined legend below all plots
        handles = list(handle_map.values())
        labels = list(handle_map.keys())
        fig.legend(handles, labels, fontsize=16, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.02))

        # Title and layout (leave space at bottom for legend)
        fig.suptitle("Parameter Evolution: All Selected Points vs Best Points (Phase Comparison)",
                     fontsize=16, y=0.995)
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])  # bottom margin for legend
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return True

    return False


def create_best_solutions_bar_plot(df_list, folder_names, output_path):
    """Create a square bar plot comparing the best total loss for each optimization technique."""

    best_solutions = []
    technique_names = []

    # Create phase labels
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

    # Extract the best solution from each dataset
    for df, phase_label in zip(df_list, phase_labels):
        if not df.empty and 'total' in df.columns:
            # Filter out invalid data
            valid_data = df[(df['total'] > 0) & (df['total'] < 1e6)]

            if not valid_data.empty:
                best_loss = valid_data['total'].min()
                best_solutions.append(best_loss)
                technique_names.append(phase_label)

    # Create square bar plot
    if best_solutions:
        fig, ax = plt.subplots(figsize=(8, 8))  # Square aspect ratio

        # Phase colors
        colors = ['orange', 'green', 'purple']

        bars = ax.bar(technique_names, best_solutions,
                      color=colors[:len(technique_names)],
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels on top of bars
        for bar, value in zip(bars, best_solutions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.2e}',
                    ha='center', va='bottom', fontsize=18, fontweight='bold')

        ax.set_xlabel('Optimization Technique', fontsize=18)
        ax.set_ylabel('Best Total Loss [W]', fontsize=18)
        ax.set_title('Best Solution Comparison Across Optimization Techniques', fontsize=16)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

        # Set tick label sizes
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    return False


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

def main_streamlined():
    """Streamlined main function that only creates the requested plot."""

    # Configuration for 3 NSGA-III folders
    folder_configs = [
        {
            'name': 'Generation-NSGA-III',
            'path': r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\run_zeta_5_Adaptive_selction_optimizer\genetic\parallel_runs',
            'method': 'generation'
        },
        {
            'name': 'simple-NSGA-III',
            'path': r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run5_diameter_zeta_simple_nsga-III\simple_nsga3',
            'method': 'standard'
        },
        {
            'name': 'Advance-NSGA-III',
            'path': r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run4_length_diameter_zeta_optimization_simple_nsga_more_generation\advanced_nsga3',
            'method': 'standard'
        }
    ]

    pressure_threshold_mpa = 10.0
    max_penalty_percent = 200.0
    num_workers = min(mp.cpu_count(), 24)

    # Load results with appropriate method for each folder
    results_list = []

    for config in folder_configs:
        try:
            if config['method'] == 'generation':
                results = load_results_generation_pattern_parallel(
                    config['path'],
                    pressure_threshold_mpa,
                    max_penalty_percent,
                    num_workers // 3
                )
            else:
                results = load_results_with_contact_pressure_parallel(
                    config['path'],
                    "NSGA-III",
                    pressure_threshold_mpa,
                    max_penalty_percent,
                    num_workers // 3
                )

            if results is None:
                results = []

        except Exception as e:
            results = []

        results_list.append(results)

    # Convert to DataFrames and validate structure
    df_list = []
    folder_names = []

    for i, (results, config) in enumerate(zip(results_list, folder_configs)):
        if results:
            df = pd.DataFrame(results)

            # Ensure consistent column naming
            if 'generation' in df.columns and 'iteration' not in df.columns:
                df['iteration'] = df['generation']

            # Prepare DataFrame for plotting
            df = prepare_dataframe_for_plotting(df, config['name'])

            df_list.append(df)
            folder_names.append(config['name'])
        else:
            df_list.append(pd.DataFrame())
            folder_names.append(config['name'])

    if all(len(df) == 0 for df in df_list):
        return

    # Create output directory and plot
    output_dir = Path("optimization_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_points_param_evolution_with_clearance_phases.png"

    # Create the specific plot
    success = create_plot_all_points_param_evolution_with_clearance_three_nsga(df_list, folder_names, output_path)
    # Create the bar plot for best solutions comparison
    bar_plot_path = output_dir / "best_solutions_comparison.png"
    bar_success = create_best_solutions_bar_plot(df_list, folder_names, bar_plot_path)

    if bar_success:
        print(f"Bar plot saved to: {bar_plot_path}")
    print_best_parameters_all_phases(df_list, folder_names)
    if success:
        return str(output_path)
    else:
        return None



if __name__ == "__main__":
    # Set start method for multiprocessing (important for Windows)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set

    result = main_streamlined()