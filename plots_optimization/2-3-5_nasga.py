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

PARAM_PATTERN_GENERATION = re.compile(
    r"T_(\d+)_(\d+)_C(\d+)of(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_dK_(\d+(?:\.\d+)?)_dZ_(\d+(?:\.\d+)?)"
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


def detect_folder_pattern_extended(folder_name: str) -> Tuple[str, Optional[re.Match]]:
    """
    Extended version that detects which pattern the folder name matches including generation pattern.

    Returns:
        Tuple of (pattern_type, match_object) where pattern_type is "new", "old", "generation", or "none"
    """
    # Try new pattern first (with CL)
    match_new = PARAM_PATTERN_NEW.search(folder_name)
    if match_new:
        return "new", match_new

    # Try old pattern (with dK)
    match_old = PARAM_PATTERN_OLD.search(folder_name)
    if match_old:
        return "old", match_old

    # Try generation pattern (T_ folders)
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

        # Extract parameters that are missing from folder name
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
        print(f"Error reading geometry file {geometry_file}: {e}")
        return {}


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


def extract_data_from_folders(base_path):
    """
    Extract data from the nested folder structure and create a CSV file.

    Args:
        base_path (str): The base path to the parallel_runs folder

    Returns:
        pandas.DataFrame: DataFrame containing extracted data
    """

    data = []
    base_path = Path(base_path)

    # Check if base path exists
    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist!")
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

    print(f"Found {len(generation_folders)} generation folders:")
    for gen_num, folder in generation_folders:
        print(f"  - Generation_{gen_num}")

    # Iterate through Generation folders in sorted order
    for generation, generation_folder in generation_folders:

        # Iterate through Individual folders
        individual_folders_found = list(generation_folder.glob("Individual_*"))
        print(f"    Found {len(individual_folders_found)} individual folders in Generation_{generation}")

        for individual_folder in individual_folders_found:
            if not individual_folder.is_dir():
                continue

            print(f"      Processing: {individual_folder.name}")

            # Extract individual info: Individual_{ind_value}_P_{phase_number}_{validation_type}_{number}C
            individual_match = re.search(r'Individual_(\d+)_P(\d+)_(Exp|Ref|Val)_(\d+)C', individual_folder.name)
            if not individual_match:
                print(f"        WARNING: Could not parse individual folder name: {individual_folder.name}")
                continue

            individual = int(individual_match.group(1))
            phase = int(individual_match.group(2))
            validation_type = individual_match.group(3)
            condition_suffix = int(individual_match.group(4))  # This captures the number before 'C'

            print(
                f"        Individual: {individual}, Phase: {phase}, Type: {validation_type}, Conditions: {condition_suffix}C")

            # Map validation type to number
            validation_mapping = {'Exp': 1, 'Ref': 2, 'Val': 3}
            validation_num = validation_mapping.get(validation_type, 0)

            # Iterate through T_ folders (condition folders)
            condition_folders_found = list(individual_folder.glob("T_*"))
            print(f"        Found {len(condition_folders_found)} condition folders")

            for condition_folder in condition_folders_found:
                if not condition_folder.is_dir():
                    continue

                # Parse T_ folder name: T_{generation_no}_{individual_no}_C_{conditionNumber}of{total_condition}_{speed}_{pressure}_{displacement}_dK_{piston_diameter}_dZ_{bushing_diameter}
                t_pattern = r'T_(\d+)_(\d+)_C(\d+)of(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_dK_(\d+(?:\.\d+)?)_dZ_(\d+(?:\.\d+)?)'
                t_match = re.search(t_pattern, condition_folder.name)

                if not t_match:
                    print(f"          WARNING: Could not parse condition folder name: {condition_folder.name}")
                    continue

                print(f"          Processing condition: {condition_folder.name}")

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
                print(
                    f"          Added record: Gen{generation}, Ind{individual}, Type{validation_type}, Condition{condition_num}")

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
        'generation': int(generation),  # Ensure scalar
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
        'iteration': int(generation),  # Map generation to iteration as scalar
        **losses,
        **contact_pressure_data,
        **geometry_params,  # Add extracted geometry parameters
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

    # Extract folder structure using the existing function
    df_structure = extract_data_from_folders(folder_path)

    if df_structure.empty:
        print(f"No valid folder structure found in {folder_path}")
        return []

    print(f"Found {len(df_structure)} condition folders to process")

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

    print(f"Processing {len(simulation_tasks)} simulations in parallel...")

    # Process simulations in parallel
    results = []
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(simulation_tasks))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_sim = {executor.submit(process_single_simulation_generation_pattern, task): task
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
    record[iter_type] = int(iter_num)  # Ensure scalar
    record["iteration"] = int(iter_num)  # Ensure scalar
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

    print(f"Completed parallel processing")

def validate_dataframe_structure(df_list, folder_names):
    """Validate DataFrame structure before plotting."""
    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty:
            print(f"\nValidating {name} DataFrame structure...")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")

            # Check if iteration column contains scalar values
            if 'iteration' in df.columns:
                sample_vals = df['iteration'].head()
                for j, val in enumerate(sample_vals):
                    if not np.isscalar(val):
                        print(f"  WARNING: iteration column contains non-scalar values at row {j}: {type(val)}")
                        print(f"  Sample value: {val}")
                        # Fix non-scalar values
                        if hasattr(val, '__iter__') and not isinstance(val, str):
                            df.loc[df.index[j], 'iteration'] = int(list(val)[0]) if len(list(val)) > 0 else 0

            # Check for required columns
            required_cols = ['dK', 'dZ', 'total', 'mechanical', 'volumetric']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  WARNING: Missing critical columns: {missing_cols}")

            # Check for optional but expected columns
            optional_cols = ['LKG', 'lF', 'zeta']
            missing_optional = [col for col in optional_cols if col not in df.columns]
            if missing_optional:
                print(f"  NOTE: Missing optional columns: {missing_optional}")
                # Add default values for missing optional columns
                for col in missing_optional:
                    if col == 'LKG':
                        df[col] = 0.0  # Default leak gap
                    elif col == 'lF':
                        df[col] = 10.0  # Default length
                    elif col == 'zeta':
                        df[col] = 1.0  # Default zeta

            # Ensure clearance column exists
            if 'clearance' not in df.columns and 'dK' in df.columns and 'dZ' in df.columns:
                df['clearance'] = df['dZ'] - df['dK']
                print(f"  Added clearance column")

            # Check data types
            numeric_cols = ['dK', 'dZ', 'total', 'mechanical', 'volumetric', 'iteration']
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        print(f"  WARNING: {col} is not numeric, attempting conversion")
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"  Validation complete for {name}")


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


def diagnose_folder_structure(folder_configs):
    """Enhanced diagnostic function to check three NSGA-III folders."""

    for config in folder_configs:
        folder_name = config['name']
        folder_path = config['path']
        method = config.get('method', 'standard')

        print(f"\n{'=' * 60}")
        print(f"DIAGNOSING {folder_name} (Method: {method})")
        print(f"{'=' * 60}")
        print(f"Path: {folder_path}")

        if not os.path.exists(folder_path):
            print("Folder does not exist!")
            continue

        if method == 'generation':
            diagnose_generation_folder_structure(folder_path)
        else:
            diagnose_standard_folder_structure(folder_path, folder_name)


def diagnose_generation_folder_structure(folder_path):
    """Diagnostic function specifically for generation pattern folders."""

    print(f"Generation pattern analysis:")
    base_folder = Path(folder_path)
    generation_folders = list(base_folder.glob("Generation_*"))
    print(f"Found {len(generation_folders)} Generation folders")

    if generation_folders:
        # Check first generation folder
        first_gen = generation_folders[0]
        individual_folders = list(first_gen.glob("Individual_*"))
        print(f"Found {len(individual_folders)} Individual folders in {first_gen.name}")

        if individual_folders:
            # Check first individual folder
            first_ind = individual_folders[0]
            condition_folders = list(first_ind.glob("T_*"))
            print(f"Found {len(condition_folders)} T_ condition folders in {first_ind.name}")

            if condition_folders:
                # Analyze pattern matching
                print(f"Pattern analysis for first 5 condition folders:")
                for i, cond in enumerate(condition_folders[:5]):
                    pattern_type, match = detect_folder_pattern_extended(cond.name)
                    print(f"  {i + 1}. {cond.name}")
                    print(f"     Pattern: {pattern_type}")

                    if pattern_type == "generation" and match:
                        print(f"     Parsed: Gen={match.group(1)}, Ind={match.group(2)}, " +
                              f"Cond={match.group(3)}, dK={match.group(8)}, dZ={match.group(9)}")

                # Check file structure
                first_cond = condition_folders[0]
                check_file_structure(first_cond)
        else:
            print("No Individual folders found")
    else:
        print("No Generation folders found - this might not be a generation pattern folder")


def diagnose_standard_folder_structure(folder_path, folder_name):
    """Diagnostic function for standard NSGA-III folder structure."""

    print(f"Standard pattern analysis:")
    base_folder = Path(folder_path)
    subfolders = [f for f in base_folder.iterdir() if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders")

    # Check first few subfolders to understand structure
    print(f"First 10 subfolders:")
    for i, folder in enumerate(subfolders[:10]):
        folder_name_str = folder.name
        print(f"  {i + 1:2d}. {folder_name_str}")

        # Check if this looks like generation folder
        if folder_name_str.startswith("Generation_G") or folder_name_str == "Initial_Sampling":
            print(f"      Valid NSGA-III generation folder")
        else:
            print(f"      Not a recognized NSGA-III generation folder")

    # Find a valid generation folder to examine simulation folders
    valid_generation_folder = None
    for folder in subfolders:
        folder_name_str = folder.name
        if folder_name_str.startswith("Generation_G") or folder_name_str == "Initial_Sampling":
            valid_generation_folder = folder
            break

    if valid_generation_folder:
        print(f"Examining simulation folders in: {valid_generation_folder.name}")
        sim_folders = [f for f in valid_generation_folder.iterdir() if f.is_dir()]
        print(f"   Found {len(sim_folders)} simulation folders")

        print(f"Pattern analysis for first 10 simulation folders:")
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

        print(f"Pattern matching summary for {valid_generation_folder.name}:")
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
                check_file_structure(matching_folder)
    else:
        print(f"No valid generation folders found for {folder_name}")


def check_file_structure(folder_path):
    """Check file structure in a simulation folder."""
    print(f"Checking file structure in: {folder_path.name}")

    # Check for required files
    required_files = [
        'input/geometry.txt',
        'input/operatingconditions.txt',
        'output/piston/piston.txt',
        'output/piston/matlab/Piston_Contact_Pressure.txt'
    ]

    for file_path in required_files:
        full_path = folder_path / file_path
        exists = full_path.exists()
        size = full_path.stat().st_size if exists else 0
        print(f"     {'Exists' if exists else 'Missing'} {file_path}: {f'({size} bytes)' if exists else ''}")


def safe_groupby_operation(df, group_col, value_col, operation='min'):
    """Safely perform groupby operations with error handling."""
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    try:
        # Ensure the grouping column contains scalar values
        if not df[group_col].apply(np.isscalar).all():
            print(f"Warning: Non-scalar values detected in {group_col}, attempting to fix...")
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
        print(f"Error in groupby operation for {group_col}, {value_col}: {e}")
        return pd.DataFrame()


def prepare_dataframe_for_plotting(df, name):
    """Prepare a DataFrame for plotting by ensuring proper data types and structure."""
    if df.empty:
        return df

    print(f"Preparing {name} DataFrame for plotting...")

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

    print(f"  {name} DataFrame prepared: {df.shape}")
    return df


def create_plot_convergence_three_nsga(args):
    """Create convergence plot for all three NSGA-III runs - for parallel execution."""
    df_list, output_dir, folder_names = args

    plt.figure(figsize=(10, 6))
    colors = ['orange', 'green', 'purple']
    markers = ['x', 's', '^']

    plot_data_exists = False

    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty and 'iteration' in df.columns and 'total' in df.columns:
            try:
                df_best = safe_groupby_operation(df, 'iteration', 'total', 'min')
                if not df_best.empty:
                    plt.plot(df_best['iteration'], df_best['total'],
                             color=colors[i], marker=markers[i], label=name, linewidth=2, markersize=6)
                    plot_data_exists = True
            except Exception as e:
                print(f"Error plotting convergence for {name}: {e}")

    if plot_data_exists:
        plt.title("Three NSGA-III Convergence Comparison", fontsize=16)
        plt.xlabel("Generation", fontsize=14)
        plt.ylabel("Best Total Loss [W]", fontsize=14)
        plt.yscale('log')
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_three_NSGA.png", dpi=300, bbox_inches='tight')
        plt.close()
        return "convergence_three_NSGA.png", "Success"

    plt.close()
    return "convergence_three_NSGA.png", "No data"


def create_plot_combined_pareto_three(args):
    """Create combined Pareto front plot for three NSGA-III runs - for parallel execution."""
    df_list, output_dir, folder_names = args

    plt.figure(figsize=(8, 6))
    colors = ['orange', 'green', 'purple']

    plot_data_exists = False

    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty and "mechanical" in df.columns and "volumetric" in df.columns:
            # Filter valid data
            valid_data = df[(df["mechanical"] > 0) & (df["volumetric"] > 0) &
                            (df["mechanical"] < 1e6) & (df["volumetric"] < 1e6)]
            if not valid_data.empty:
                plt.scatter(valid_data["mechanical"], valid_data["volumetric"],
                            color=colors[i], alpha=0.6, label=name, s=30)
                plot_data_exists = True

    if plot_data_exists:
        plt.xlabel("Mechanical Loss [W]", fontsize=14)
        plt.ylabel("Volumetric Loss [W]", fontsize=14)
        plt.title("Three NSGA-III: Mechanical vs Volumetric Loss", fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "combined_pareto_three_NSGA.png", dpi=300, bbox_inches='tight')
        plt.close()
        return "combined_pareto_three_NSGA.png", "Success"

    plt.close()
    return "combined_pareto_three_NSGA.png", "No data"


def create_plot_param_evolution_three_nsga(args):
    """Create parameter evolution plot for three NSGA-III runs - for parallel execution."""
    df_list, output_dir, folder_names = args

    # Prepare best iteration data for all three
    df_best_list = []
    for df in df_list:
        if not df.empty and 'iteration' in df.columns and 'total' in df.columns:
            try:
                # Get best solution for each iteration
                best_indices = df.groupby('iteration')['total'].idxmin()
                df_best = df.loc[best_indices].copy()

                # Ensure clearance exists
                if 'clearance' not in df_best.columns and 'dZ' in df_best.columns and 'dK' in df_best.columns:
                    df_best['clearance'] = df_best['dZ'] - df_best['dK']

                df_best_list.append(df_best)
            except Exception as e:
                print(f"Error preparing data for param evolution: {e}")
                df_best_list.append(pd.DataFrame())
        else:
            df_best_list.append(pd.DataFrame())

    parameters_with_clearance = ["clearance", "LKG", "lF", "zeta"]
    plots_to_show = parameters_with_clearance + ["convergence"]

    if any(not df.empty for df in df_best_list):
        fig, axes = plt.subplots(len(plots_to_show), 1, figsize=(12, 20), sharex=True)
        if len(plots_to_show) == 1:
            axes = [axes]

        colors = ['orange', 'green', 'purple']
        markers = ['x', 's', '^']
        linestyles = ['-', '--', '-.']

        for i, param in enumerate(plots_to_show):
            if param == "convergence":
                # Special handling for convergence plot
                for j, (df, name) in enumerate(zip(df_list, folder_names)):
                    if not df.empty and 'iteration' in df.columns and 'total' in df.columns:
                        try:
                            df_conv = safe_groupby_operation(df, 'iteration', 'total', 'min')
                            if not df_conv.empty:
                                axes[i].plot(df_conv['iteration'], df_conv['total'],
                                             marker=markers[j], linestyle=linestyles[j], color=colors[j],
                                             label=name, linewidth=2, markersize=6)
                        except Exception as e:
                            print(f"Error plotting convergence for {name}: {e}")

                axes[i].set_ylabel("Best Total Power Loss [W]", fontsize=14)
                axes[i].set_yscale('log')
            else:
                # Regular parameter plots
                for j, (df_best, name) in enumerate(zip(df_best_list, folder_names)):
                    if not df_best.empty and param in df_best.columns and 'iteration' in df_best.columns:
                        try:
                            axes[i].plot(df_best['iteration'], df_best[param],
                                         marker=markers[j], linestyle=linestyles[j], color=colors[j],
                                         label=name, linewidth=2, markersize=6)
                        except Exception as e:
                            print(f"Error plotting {param} for {name}: {e}")

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
            axes[i].legend(fontsize=12)

        axes[-1].set_xlabel("Generation", fontsize=14)
        axes[-1].tick_params(axis='both', labelsize=12)

        fig.suptitle("Three NSGA-III Parameter Evolution Comparison", fontsize=16, y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / "param_evolution_three_NSGA.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        return "param_evolution_three_NSGA.png", "Success"

    return "param_evolution_three_NSGA.png", "No data"


def create_plot_contact_pressure_comparison_three(args):
    """Create contact pressure analysis for three NSGA-III runs - for parallel execution."""
    df_list, output_dir, folder_names, pressure_threshold_mpa = args

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['orange', 'green', 'purple']

    # Individual scatter plots for each run
    plot_count = 0
    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty and 'total' in df.columns and 'max_contact_pressure' in df.columns:
            if plot_count < 3:  # Only plot first 3 runs
                if plot_count < 2:
                    ax = axes[0, plot_count]
                else:
                    ax = axes[1, 0]

                # Filter valid data
                valid_data = df[(df['total'] > 0) & (df['total'] < 1e6) &
                                (df['max_contact_pressure'] >= 0)]

                if not valid_data.empty and 'contact_pressure_penalty' in valid_data.columns:
                    scatter = ax.scatter(valid_data["total"], valid_data["max_contact_pressure"],
                                         c=valid_data["contact_pressure_penalty"] * 100,
                                         alpha=0.7, s=50, cmap='viridis', vmin=0, vmax=200)
                    fig.colorbar(scatter, ax=ax, label='Penalty (%)')

                ax.set_xlabel("Total Loss [W]", fontsize=12)
                ax.set_ylabel("Max Contact Pressure [MPa]", fontsize=12)
                ax.set_title(f"{name}: Contact Pressure vs Total Loss", fontsize=14)
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=pressure_threshold_mpa, color='red', linestyle='--', alpha=0.7)

                plot_count += 1

    # Combined comparison plot
    ax_combined = axes[1, 1]
    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty and 'total' in df.columns and 'max_contact_pressure' in df.columns:
            # Filter valid data
            valid_data = df[(df['total'] > 0) & (df['total'] < 1e6) &
                            (df['max_contact_pressure'] >= 0)]

            if not valid_data.empty:
                ax_combined.scatter(valid_data["total"], valid_data["max_contact_pressure"],
                                    color=colors[i], alpha=0.6, s=30, label=name)

    ax_combined.axhline(y=pressure_threshold_mpa, color='red', linestyle='--',
                        label=f'Pressure Threshold ({pressure_threshold_mpa} MPa)', linewidth=2)
    ax_combined.set_xlabel("Total Loss [W]", fontsize=12)
    ax_combined.set_ylabel("Max Contact Pressure [MPa]", fontsize=12)
    ax_combined.set_title("Combined: Contact Pressure vs Total Loss", fontsize=14)
    ax_combined.set_xscale('log')
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "contact_pressure_comparison_three_NSGA.png", dpi=300, bbox_inches='tight')
    plt.close()
    return "contact_pressure_comparison_three_NSGA.png", "Success"


def create_plot_penalized_convergence_three_nsga(args):
    """Create penalized vs non-penalized convergence comparison for three NSGA-III runs."""
    df_list, output_dir, folder_names = args

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    colors = ['orange', 'green', 'purple']
    markers = ['x', 's', '^']
    linestyles = ['-', '--', '-.']

    plot_data_exists = False

    # Top plot: Total loss convergence
    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty and 'iteration' in df.columns and 'total' in df.columns:
            try:
                df_conv = safe_groupby_operation(df, 'iteration', 'total', 'min')
                if not df_conv.empty:
                    axes[0].plot(df_conv['iteration'], df_conv['total'],
                                 marker=markers[i], linestyle=linestyles[i], color=colors[i],
                                 label=f'{name} (Total Loss)', linewidth=2, markersize=6)
                    plot_data_exists = True
            except Exception as e:
                print(f"Error plotting total loss convergence for {name}: {e}")

    axes[0].set_ylabel("Best Total Loss [W]", fontsize=14)
    axes[0].set_yscale('log')
    axes[0].set_title("Total Loss Convergence", fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=12)

    # Bottom plot: Penalized loss convergence
    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty and 'iteration' in df.columns and 'penalized_total' in df.columns:
            try:
                df_conv_penalty = safe_groupby_operation(df, 'iteration', 'penalized_total', 'min')
                if not df_conv_penalty.empty:
                    axes[1].plot(df_conv_penalty['iteration'], df_conv_penalty['penalized_total'],
                                 marker=markers[i], linestyle=linestyles[i], color=colors[i],
                                 label=f'{name} (Penalized)', linewidth=2, markersize=6)
            except Exception as e:
                print(f"Error plotting penalized loss convergence for {name}: {e}")

    axes[1].set_xlabel("Generation", fontsize=14)
    axes[1].set_ylabel("Best Penalized Loss [W]", fontsize=14)
    axes[1].set_yscale('log')
    axes[1].set_title("Penalized Loss Convergence", fontsize=16)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=12)

    if plot_data_exists:
        plt.tight_layout()
        plt.savefig(output_dir / "penalized_convergence_three_NSGA.png", dpi=300, bbox_inches='tight')
        plt.close()
        return "penalized_convergence_three_NSGA.png", "Success"

    plt.close()
    return "penalized_convergence_three_NSGA.png", "No data"


def create_plot_all_points_param_evolution_with_clearance_three_nsga(args):
    """Create parameter evolution plot showing ALL points for three NSGA-III runs with best markers."""
    df_list, output_dir, folder_names = args

    # Create phase labels based on folder order: Folder 1 = Phase I, etc.
    phase_labels = []
    for i, folder_name in enumerate(folder_names):
        if i == 0:
            phase_labels.append("Phase I")
        elif i == 1:
            phase_labels.append("Phase II")
        elif i == 2:
            phase_labels.append("Phase III")
        else:
            phase_labels.append(f"Phase {i + 1}")  # For any additional folders

    # Add clearance calculation for all dataframes
    processed_df_list = []
    for df in df_list:
        if not df.empty:
            df = df.copy()
            if 'clearance' not in df.columns and 'dK' in df.columns and 'dZ' in df.columns:
                df['clearance'] = df['dZ'] - df['dK']
        processed_df_list.append(df)

    # Find best overall solutions (across all iterations/generations)
    best_overall_list = []
    for df in processed_df_list:
        if not df.empty and 'total' in df.columns:
            best_overall_idx = df['total'].idxmin()
            best_overall_list.append(df.loc[best_overall_idx])
        else:
            best_overall_list.append(None)

    parameters_with_clearance = ["clearance", "LKG", "lF", "zeta"]
    plots_to_show = parameters_with_clearance + ["convergence"]

    colors = ['orange', 'green', 'purple']
    markers = ['o', '^', 's']

    if any(not df.empty for df in processed_df_list):
        fig, axes = plt.subplots(len(plots_to_show), 1, figsize=(12, 20), sharex=True)
        if len(plots_to_show) == 1:
            axes = [axes]

        for i, param in enumerate(plots_to_show):
            if param == "convergence":
                # Special handling for convergence plot
                for j, (df, phase_label) in enumerate(zip(processed_df_list, phase_labels)):
                    if not df.empty and 'iteration' in df.columns and 'total' in df.columns:
                        try:
                            df_conv = safe_groupby_operation(df, 'iteration', 'total', 'min')
                            if not df_conv.empty:
                                axes[i].plot(df_conv['iteration'], df_conv['total'],
                                             marker=markers[j], linestyle='-', color=colors[j],
                                             label=f'{phase_label} Best', linewidth=2, markersize=6)

                            # Add marker for best overall solution
                            if best_overall_list[j] is not None:
                                axes[i].scatter([best_overall_list[j]['iteration']], [best_overall_list[j]['total']],
                                                s=200, color=['red', 'darkred', 'maroon'][j], marker='*',
                                                edgecolors='black', linewidth=2,
                                                label=f'Best {phase_label} Overall', zorder=10)
                        except Exception as e:
                            print(f"Error plotting convergence for {phase_label}: {e}")

                axes[i].set_ylabel("Best Total Power Loss [W]", fontsize=14)
                axes[i].set_yscale('log')
            else:
                # Scatter plot showing ALL points for each iteration/generation
                for j, (df, phase_label) in enumerate(zip(processed_df_list, phase_labels)):
                    if not df.empty and param in df.columns and 'iteration' in df.columns:
                        try:
                            # Add some jitter for better visibility
                            jitter = np.random.normal(0, 0.15, len(df))
                            axes[i].scatter(df['iteration'] + jitter, df[param],
                                            alpha=0.6, s=20, color=colors[j],
                                            label=f'{phase_label} All Points',
                                            edgecolors='none', marker=markers[j])

                            # Overlay the best points for each iteration
                            best_indices = df.groupby('iteration')['total'].idxmin()
                            df_best = df.loc[best_indices]
                            axes[i].plot(df_best['iteration'], df_best[param],
                                         marker=markers[j], linestyle='-', color=colors[j],
                                         label=f'{phase_label} Best per Generation',
                                         linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1)

                            # Add marker for best overall solution
                            if best_overall_list[j] is not None and param in best_overall_list[j]:
                                axes[i].scatter([best_overall_list[j]['iteration']], [best_overall_list[j][param]],
                                                s=200, color=['red', 'darkred', 'maroon'][j], marker='*',
                                                edgecolors='black', linewidth=2,
                                                label=f'Best {phase_label} Overall', zorder=10)
                        except Exception as e:
                            print(f"Error plotting {param} for {phase_label}: {e}")

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

        axes[-1].set_xlabel("Generation", fontsize=14)
        axes[-1].tick_params(axis='both', labelsize=12)

        fig.suptitle("Parameter Evolution: All Selected Points vs Best Points (Phase Comparison)", fontsize=16, y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / "all_points_param_evolution_with_clearance_phases.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        return "all_points_param_evolution_with_clearance_phases.png", "Success"

    return "all_points_param_evolution_with_clearance_phases.png", "No data"


# Individual plotting functions for each NSGA-III run

def create_plot_param_distribution_nsga3(args):
    """Create NSGA-III parameter distribution plots - for parallel execution."""
    df_nsga3, output_dir, parameters = args
    if df_nsga3.empty:
        return "param_distribution_NSGA3.png", "No data"

    # Filter parameters that exist in the dataframe
    available_params = [p for p in parameters if p in df_nsga3.columns]
    if not available_params:
        return "param_distribution_NSGA3.png", "No valid parameters"

    fig, axes = plt.subplots(len(available_params), 1, figsize=(8, 2 * len(available_params)))
    if len(available_params) == 1:
        axes = [axes]

    for i, param in enumerate(available_params):
        try:
            sns.histplot(df_nsga3, x=param, kde=True, ax=axes[i], color='orange', bins=20)
            axes[i].set_title(f"NSGA-III: Distribution of {param}")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Count")
        except Exception as e:
            print(f"Error plotting distribution for {param}: {e}")
            axes[i].text(0.5, 0.5, f"Error plotting {param}", ha='center', va='center', transform=axes[i].transAxes)

    fig.tight_layout()
    fig.savefig(output_dir / "param_distribution_NSGA3.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return "param_distribution_NSGA3.png", "Success"


def create_plot_param_vs_loss_nsga3(args):
    """Create NSGA-III parameter vs loss scatter plots - for parallel execution."""
    df_nsga3, output_dir, parameters = args
    if df_nsga3.empty or 'total' not in df_nsga3.columns:
        return "param_vs_loss_NSGA3.png", "No data or missing total column"

    # Filter parameters that exist in the dataframe
    available_params = [p for p in parameters if p in df_nsga3.columns]
    if not available_params:
        return "param_vs_loss_NSGA3.png", "No valid parameters"

    fig, axes = plt.subplots(len(available_params), 1, figsize=(8, 3 * len(available_params)))
    if len(available_params) == 1:
        axes = [axes]

    for i, param in enumerate(available_params):
        try:
            # Filter valid data
            valid_data = df_nsga3[(df_nsga3[param].notna()) & (df_nsga3['total'] > 0) & (df_nsga3['total'] < 1e6)]

            if not valid_data.empty:
                axes[i].scatter(valid_data[param], valid_data["total"], color='orange', alpha=0.6)
                axes[i].set_title(f"NSGA-III: {param} vs Total Loss")
                axes[i].set_xlabel(param)
                axes[i].set_ylabel("Total Loss")
                axes[i].set_yscale('log')
                axes[i].grid(True)
            else:
                axes[i].text(0.5, 0.5, f"No valid data for {param}", ha='center', va='center',
                             transform=axes[i].transAxes)
        except Exception as e:
            print(f"Error plotting {param} vs loss: {e}")
            axes[i].text(0.5, 0.5, f"Error plotting {param}", ha='center', va='center', transform=axes[i].transAxes)

    fig.tight_layout()
    fig.savefig(output_dir / "param_vs_loss_NSGA3.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return "param_vs_loss_NSGA3.png", "Success"


def create_plot_pairplot_nsga3(args):
    """Create NSGA-III pairplot - for parallel execution."""
    df_nsga3, output_dir, parameters = args
    if df_nsga3.empty:
        return "pairplot_NSGA3.png", "No data"

    # Filter parameters that exist in the dataframe and add total
    available_params = [p for p in parameters if p in df_nsga3.columns]
    if 'total' in df_nsga3.columns:
        cols_for_pairplot = available_params + ["total"]
    else:
        cols_for_pairplot = available_params

    if len(cols_for_pairplot) < 2:
        return "pairplot_NSGA3.png", "Not enough valid columns for pairplot"

    try:
        df_plot = df_nsga3[cols_for_pairplot].copy()

        # Remove invalid data
        df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_plot) > 5000:
            df_plot = df_plot.sample(n=5000, random_state=42)

        if len(df_plot) < 10:
            return "pairplot_NSGA3.png", "Not enough valid data points"

        sns.pairplot(df_plot, diag_kind="kde")
        plt.suptitle("NSGA-III Pairwise Parameter Relationships", y=1.02)
        plt.savefig(output_dir / "pairplot_NSGA3.png", dpi=300, bbox_inches='tight')
        plt.close()
        return "pairplot_NSGA3.png", "Success"
    except Exception as e:
        print(f"Error creating pairplot: {e}")
        return "pairplot_NSGA3.png", f"Error: {e}"


def create_plot_pareto_nsga3(args):
    """Create NSGA-III Pareto front plot - for parallel execution."""
    df_nsga3, output_dir = args
    if df_nsga3.empty or "mechanical" not in df_nsga3.columns or "volumetric" not in df_nsga3.columns:
        return "pareto_NSGA3.png", "No data or missing columns"

    try:
        # Filter valid data
        valid_data = df_nsga3[(df_nsga3["mechanical"] > 0) & (df_nsga3["volumetric"] > 0) &
                              (df_nsga3["mechanical"] < 1e6) & (df_nsga3["volumetric"] < 1e6)]

        if valid_data.empty:
            return "pareto_NSGA3.png", "No valid data points"

        plt.figure(figsize=(6, 5))
        plt.scatter(valid_data["mechanical"], valid_data["volumetric"], color='orange', alpha=0.7)
        plt.title("NSGA-III: Mechanical vs Volumetric Loss")
        plt.xlabel("Mechanical Loss")
        plt.ylabel("Volumetric Loss")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "pareto_NSGA3.png", dpi=300, bbox_inches='tight')
        plt.close()
        return "pareto_NSGA3.png", "Success"
    except Exception as e:
        print(f"Error creating Pareto plot: {e}")
        return "pareto_NSGA3.png", f"Error: {e}"


def create_sensitivity_analysis_three_nsga_parallel(df_list, folder_names, output_dir):
    """Create sensitivity analysis for three NSGA-III runs with parallel correlation calculations."""

    def calculate_correlations(df, parameters):
        """Calculate correlations for a single dataset."""
        correlations = {}
        for param in parameters:
            if not df.empty and param in df.columns and 'total' in df.columns:
                try:
                    # Filter valid data
                    valid_data = df[(df[param].notna()) & (df['total'].notna()) &
                                    (df['total'] > 0) & (df['total'] < 1e6)]
                    if len(valid_data) > 10:
                        corr, _ = pearsonr(valid_data[param], valid_data["total"])
                        correlations[param] = abs(corr)
                    else:
                        correlations[param] = np.nan
                except:
                    correlations[param] = np.nan
            else:
                correlations[param] = np.nan
        return correlations

    # Add clearance calculation for all datasets
    for i, df in enumerate(df_list):
        if not df.empty and 'dK' in df.columns and 'dZ' in df.columns:
            if 'clearance' not in df.columns:
                df_list[i] = df.copy()
                df_list[i]["clearance"] = df["dZ"] - df["dK"]

    selected_params = ["clearance", "zeta", "lF", "LKG"]
    param_labels = {"clearance": "clearance", "zeta": "γ", "lF": "lF", "LKG": "LKG"}

    # Calculate correlations in parallel for all three runs
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for df in df_list:
            future = executor.submit(calculate_correlations, df, selected_params)
            futures.append(future)

        all_correlations = []
        for future in futures:
            all_correlations.append(future.result())

    # Prepare data for plotting
    sensitivity_data = []
    for param in selected_params:
        data_row = {"parameter": param_labels[param]}
        for i, (correlations, name) in enumerate(zip(all_correlations, folder_names)):
            data_row[name] = correlations[param]
        sensitivity_data.append(data_row)

    df_sensitivity = pd.DataFrame(sensitivity_data)

    # Create plot
    x = np.arange(len(df_sensitivity["parameter"]))
    width = 0.25
    colors = ['orange', 'green', 'purple']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(folder_names):
        if name in df_sensitivity.columns:
            offset = (i - 1) * width
            values = df_sensitivity[name].fillna(0)  # Replace NaN with 0
            bars = ax.bar(x + offset, values, width,
                          label=name, color=colors[i], alpha=0.8)

    ax.set_xlabel("Parameter", fontsize=16)
    ax.set_ylabel("Sensitivity (|Pearson Correlation|)", fontsize=16)
    ax.set_title("Parameter Sensitivity Comparison - Three NSGA-III Runs", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sensitivity["parameter"], fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_bar_chart_three_NSGA.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return "sensitivity_bar_chart_three_NSGA.png"


def create_comprehensive_comparison_plots_three_nsga(df_list, folder_names, output_dir):
    """Create comprehensive comparison plots for the three NSGA-III runs."""

    # Overall comparison statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = ['orange', 'green', 'purple']

    # Statistical comparison
    stats_data = []
    for df, name in zip(df_list, folder_names):
        if not df.empty and 'total' in df.columns:
            valid_data = df[(df['total'] > 0) & (df['total'] < 1e6)]

            if not valid_data.empty:
                stats_data.append({
                    'name': name,
                    'min_total_loss': valid_data['total'].min(),
                    'mean_total_loss': valid_data['total'].mean(),
                    'min_contact_pressure': valid_data.get('max_contact_pressure', pd.Series([0])).min(),
                    'mean_contact_pressure': valid_data.get('max_contact_pressure', pd.Series([0])).mean(),
                    'mean_penalty': valid_data.get('contact_pressure_penalty', pd.Series([0])).mean() * 100
                })

    if stats_data:
        df_stats = pd.DataFrame(stats_data)

        # Plot 1: Min total loss comparison
        axes[0, 0].bar(df_stats['name'], df_stats['min_total_loss'], color=colors[:len(df_stats)])
        axes[0, 0].set_title('Minimum Total Loss Comparison')
        axes[0, 0].set_ylabel('Min Total Loss [W]')
        axes[0, 0].set_yscale('log')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Mean total loss comparison
        axes[0, 1].bar(df_stats['name'], df_stats['mean_total_loss'], color=colors[:len(df_stats)])
        axes[0, 1].set_title('Average Total Loss Comparison')
        axes[0, 1].set_ylabel('Mean Total Loss [W]')
        axes[0, 1].set_yscale('log')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Contact pressure comparison
        axes[1, 0].bar(df_stats['name'], df_stats['mean_contact_pressure'], color=colors[:len(df_stats)])
        axes[1, 0].set_title('Average Contact Pressure Comparison')
        axes[1, 0].set_ylabel('Mean Contact Pressure [MPa]')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 4: Penalty comparison
        axes[1, 1].bar(df_stats['name'], df_stats['mean_penalty'], color=colors[:len(df_stats)])
        axes[1, 1].set_title('Average Penalty Comparison')
        axes[1, 1].set_ylabel('Mean Penalty [%]')
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_stats_comparison_three_NSGA.png", dpi=300, bbox_inches='tight')
    plt.close()

    return "comprehensive_stats_comparison_three_NSGA.png"


def create_statistical_comparison_three_nsga(df_list, folder_names, output_dir):
    """Create detailed statistical comparison between the three NSGA-III runs."""

    # Prepare statistical data
    stats_summary = []

    for df, name in zip(df_list, folder_names):
        if not df.empty:
            # Add clearance calculation
            if 'clearance' not in df.columns and 'dK' in df.columns and 'dZ' in df.columns:
                df['clearance'] = df['dZ'] - df['dK']

            # Filter valid data
            valid_data = df[(df.get('total', pd.Series([])) > 0) & (df.get('total', pd.Series([])) < 1e6)]

            if not valid_data.empty:
                # Calculate statistics
                stats = {
                    'Run': name,
                    'Total_Simulations': len(df),
                    'Valid_Simulations': len(valid_data),
                    'Valid_Pressure_Data': len(df[df.get('contact_pressure_valid',
                                                         False) == True]) if 'contact_pressure_valid' in df.columns else 0,
                    'Min_Total_Loss': valid_data['total'].min() if 'total' in valid_data.columns else np.nan,
                    'Mean_Total_Loss': valid_data['total'].mean() if 'total' in valid_data.columns else np.nan,
                    'Std_Total_Loss': valid_data['total'].std() if 'total' in valid_data.columns else np.nan,
                    'Min_Penalized_Loss': valid_data.get('penalized_total', pd.Series([np.nan])).min(),
                    'Mean_Penalized_Loss': valid_data.get('penalized_total', pd.Series([np.nan])).mean(),
                    'Min_Contact_Pressure': valid_data.get('max_contact_pressure', pd.Series([np.nan])).min(),
                    'Max_Contact_Pressure': valid_data.get('max_contact_pressure', pd.Series([np.nan])).max(),
                    'Mean_Contact_Pressure': valid_data.get('max_contact_pressure', pd.Series([np.nan])).mean(),
                    'Mean_Penalty_Percent': valid_data.get('contact_pressure_penalty', pd.Series([0])).mean() * 100,
                    'Max_Penalty_Percent': valid_data.get('contact_pressure_penalty', pd.Series([0])).max() * 100,
                    'Mean_Clearance': valid_data.get('clearance', pd.Series([np.nan])).mean(),
                    'Std_Clearance': valid_data.get('clearance', pd.Series([np.nan])).std(),
                    'Min_Clearance': valid_data.get('clearance', pd.Series([np.nan])).min(),
                    'Max_Clearance': valid_data.get('clearance', pd.Series([np.nan])).max()
                }
                stats_summary.append(stats)

    # Save statistical summary
    if stats_summary:
        df_stats = pd.DataFrame(stats_summary)
        df_stats.to_csv(output_dir / "statistical_comparison_three_nsga.csv", index=False)

        # Create statistical comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        colors = ['orange', 'green', 'purple']

        # Plot 1: Total simulations and valid data
        ax = axes[0, 0]
        x_pos = range(len(df_stats))
        width = 0.35
        ax.bar([p - width / 2 for p in x_pos], df_stats['Total_Simulations'], width,
               label='Total Simulations', color=colors, alpha=0.8)
        ax.bar([p + width / 2 for p in x_pos], df_stats['Valid_Simulations'], width,
               label='Valid Simulations', color=colors, alpha=0.6)
        ax.set_xlabel('NSGA-III Runs')
        ax.set_ylabel('Count')
        ax.set_title('Simulation Count Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_stats['Run'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss comparison
        ax = axes[0, 1]
        valid_losses = df_stats['Min_Total_Loss'].dropna()
        if not valid_losses.empty:
            ax.bar(df_stats['Run'][:len(valid_losses)], valid_losses, color=colors[:len(valid_losses)], alpha=0.8)
            ax.set_xlabel('NSGA-III Runs')
            ax.set_ylabel('Min Total Loss [W]')
            ax.set_title('Minimum Total Loss Comparison')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        # Plot 3: Contact pressure comparison
        ax = axes[0, 2]
        valid_pressures = df_stats['Mean_Contact_Pressure'].dropna()
        if not valid_pressures.empty:
            ax.bar(df_stats['Run'][:len(valid_pressures)], valid_pressures, color=colors[:len(valid_pressures)],
                   alpha=0.8)
            ax.set_xlabel('NSGA-III Runs')
            ax.set_ylabel('Mean Contact Pressure [MPa]')
            ax.set_title('Average Contact Pressure Comparison')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        # Plot 4: Penalty comparison
        ax = axes[1, 0]
        valid_penalties = df_stats['Mean_Penalty_Percent'].dropna()
        if not valid_penalties.empty:
            ax.bar(df_stats['Run'][:len(valid_penalties)], valid_penalties, color=colors[:len(valid_penalties)],
                   alpha=0.8)
            ax.set_xlabel('NSGA-III Runs')
            ax.set_ylabel('Mean Penalty [%]')
            ax.set_title('Average Penalty Comparison')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        # Plot 5: Clearance comparison
        ax = axes[1, 1]
        valid_clearances = df_stats['Mean_Clearance'].dropna()
        if not valid_clearances.empty:
            ax.bar(df_stats['Run'][:len(valid_clearances)], valid_clearances, color=colors[:len(valid_clearances)],
                   alpha=0.8)
            ax.set_xlabel('NSGA-III Runs')
            ax.set_ylabel('Mean Clearance [μm]')
            ax.set_title('Average Clearance Comparison')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        # Plot 6: Performance improvement comparison
        ax = axes[1, 2]
        valid_losses_for_improvement = df_stats['Min_Total_Loss'].dropna()
        if len(valid_losses_for_improvement) > 1:
            improvement_ratios = []
            baseline_loss = valid_losses_for_improvement.iloc[0]  # Use first run as baseline
            for loss in valid_losses_for_improvement:
                if baseline_loss > 0:
                    improvement = ((baseline_loss - loss) / baseline_loss) * 100
                else:
                    improvement = 0
                improvement_ratios.append(improvement)

            bars = ax.bar(df_stats['Run'][:len(improvement_ratios)], improvement_ratios,
                          color=colors[:len(improvement_ratios)], alpha=0.8)
            ax.set_xlabel('NSGA-III Runs')
            ax.set_ylabel('Improvement vs Baseline [%]')
            ax.set_title('Performance Improvement Comparison')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)

            # Add value labels on bars
            for bar, value in zip(bars, improvement_ratios):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + (max(improvement_ratios) * 0.01),
                        f'{value:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / "statistical_comparison_plots_three_nsga.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("Statistical comparison saved to statistical_comparison_three_nsga.csv")
        print("Statistical comparison plots saved")

        return "statistical_comparison_plots_three_nsga.png"

    return "No data"


def create_all_plots_parallel_complete_three_nsga(df_list, folder_names, output_dir, pressure_threshold_mpa=100.0,
                                                  max_workers=None):
    """Create ALL plots for three NSGA-III runs in parallel."""

    # Define all plotting tasks for three folders
    plot_functions = [
        # Main comparison plots
        (create_plot_convergence_three_nsga, (df_list, output_dir, folder_names)),
        (create_plot_combined_pareto_three, (df_list, output_dir, folder_names)),
        (create_plot_param_evolution_three_nsga, (df_list, output_dir, folder_names)),
        (create_plot_contact_pressure_comparison_three, (df_list, output_dir, folder_names, pressure_threshold_mpa)),
        (create_plot_penalized_convergence_three_nsga, (df_list, output_dir, folder_names)),
        # Add the new all points plot with best markers
        (create_plot_all_points_param_evolution_with_clearance_three_nsga, (df_list, output_dir, folder_names)),
    ]

    # Add individual plots for each NSGA-III run
    parameters = ["dK", "dZ", "LKG", "lF", "zeta"]
    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty:
            # Create filename suffixes to distinguish between runs
            individual_output_dir = output_dir / f"individual_plots_{name.replace('-', '_').replace(' ', '_')}"
            individual_output_dir.mkdir(exist_ok=True)

            plot_functions.extend([
                (create_plot_param_distribution_nsga3, (df, individual_output_dir, parameters)),
                (create_plot_param_vs_loss_nsga3, (df, individual_output_dir, parameters)),
                (create_plot_pairplot_nsga3, (df, individual_output_dir, parameters)),
                (create_plot_pareto_nsga3, (df, individual_output_dir))
            ])

    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(plot_functions))

    print(f"Creating {len(plot_functions)} plots in parallel with {max_workers} workers...")

    # Use ThreadPoolExecutor for plotting
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

def save_optimization_results_three_nsga_parallel(df_list, folder_names, output_dir, top_n_save=5):
    """Save optimization results for three NSGA-III runs in parallel."""
    columns_to_save = ["optimizer", "iteration", "total", "penalized_total", "mechanical", "volumetric",
                       "max_contact_pressure", "avg_contact_pressure", "contact_pressure_penalty",
                       "dK", "dZ", "clearance", "LKG", "lF", "zeta", "pattern_type", "contact_pressure_valid"]

    def prepare_top_results(df, optimizer_name, top_n, penalty_col='penalized_total', loss_col='total'):
        """Prepare top results for a single optimizer."""
        results = []
        if not df.empty:
            # Ensure required columns exist
            for col in ['clearance', 'penalized_total']:
                if col not in df.columns:
                    if col == 'clearance' and 'dK' in df.columns and 'dZ' in df.columns:
                        df[col] = df['dZ'] - df['dK']
                    elif col == 'penalized_total' and 'total' in df.columns:
                        df[col] = df['total']

            # Filter valid columns that exist in the dataframe
            available_columns = [col for col in columns_to_save if col in df.columns]

            # Filter valid data
            valid_data = df[(df.get('total', pd.Series([])) > 0) & (df.get('total', pd.Series([])) < 1e6)]

            if not valid_data.empty:
                # Top results with penalty
                if penalty_col in valid_data.columns:
                    top_penalty = valid_data.nsmallest(top_n, penalty_col).copy()
                    top_penalty["optimizer"] = optimizer_name
                    results.append(top_penalty[available_columns])

                # Top results without penalty
                if loss_col in valid_data.columns:
                    top_no_penalty = valid_data.nsmallest(top_n, loss_col).copy()
                    top_no_penalty["optimizer"] = f"{optimizer_name}_no_penalty"
                    results.append(top_no_penalty[available_columns])

        return results

    # Prepare results in parallel for all three runs
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for df, name in zip(df_list, folder_names):
            future = executor.submit(prepare_top_results, df, name, top_n_save)
            futures.append(future)

        all_results = []
        for future in futures:
            all_results.extend(future.result())

    # Combine and save results
    if all_results:
        df_combined = pd.concat(all_results, ignore_index=True)
        df_combined.to_csv(output_dir / "top_optimal_results_three_NSGA_with_contact_penalty.csv", index=False)
        print(f"Saved combined results to: top_optimal_results_three_NSGA_with_contact_penalty.csv")

    # Also save individual results for each run
    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty:
            individual_results = prepare_top_results(df, name, top_n_save)
            if individual_results:
                df_individual = pd.concat(individual_results, ignore_index=True)
                filename = f"top_optimal_results_{name.replace('-', '_').replace(' ', '_')}.csv"
                df_individual.to_csv(output_dir / filename, index=False)
                print(f"Saved {name} results to: {filename}")


def print_detailed_best_solutions_three(df_list, folder_names):
    """Print detailed comparison of best solutions for three NSGA-III runs."""
    print(f"\nBEST SOLUTIONS COMPARISON (THREE NSGA-III RUNS):")
    print("=" * 80)

    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty and 'total' in df.columns:
            # Filter valid data
            valid_data = df[(df['total'] > 0) & (df['total'] < 1e6)]

            if not valid_data.empty:
                # Best with penalty
                best_penalty_idx = None
                best_penalty = None
                if 'penalized_total' in valid_data.columns:
                    best_penalty_idx = valid_data['penalized_total'].idxmin()
                    best_penalty = valid_data.loc[best_penalty_idx]

                # Best without penalty
                best_no_penalty_idx = valid_data['total'].idxmin()
                best_no_penalty = valid_data.loc[best_no_penalty_idx]

                print(f"\n{name} BEST SOLUTIONS:")

                if best_penalty is not None:
                    print(f"   WITH PENALTY (Generation {best_penalty.get('iteration', 'N/A')}):")
                    print(f"      - Total Loss: {best_penalty['total']:.4e} W")
                    print(f"      - Penalized Total: {best_penalty.get('penalized_total', 'N/A'):.4e} W" if isinstance(
                        best_penalty.get('penalized_total'), (int, float)) else "      - Penalized Total: N/A")
                    print(f"      - Max Contact Pressure: {best_penalty.get('max_contact_pressure', 0):.1f} MPa")
                    print(f"      - Penalty: {best_penalty.get('contact_pressure_penalty', 0) * 100:.1f}%")
                    print(
                        f"      - Parameters: dK={best_penalty.get('dK', 0):.3f}, dZ={best_penalty.get('dZ', 0):.3f}, clearance={best_penalty.get('clearance', 0):.3f}")

                print(f"   WITHOUT PENALTY (Generation {best_no_penalty.get('iteration', 'N/A')}):")
                print(f"      - Total Loss: {best_no_penalty['total']:.4e} W")
                print(f"      - Penalized Total: {best_no_penalty.get('penalized_total', 'N/A'):.4e} W" if isinstance(
                    best_no_penalty.get('penalized_total'), (int, float)) else "      - Penalized Total: N/A")
                print(f"      - Max Contact Pressure: {best_no_penalty.get('max_contact_pressure', 0):.1f} MPa")
                print(f"      - Penalty: {best_no_penalty.get('contact_pressure_penalty', 0) * 100:.1f}%")
                print(
                    f"      - Parameters: dK={best_no_penalty.get('dK', 0):.3f}, dZ={best_no_penalty.get('dZ', 0):.3f}, clearance={best_no_penalty.get('clearance', 0):.3f}")


def print_top_results_summary_three_nsga(df_list, folder_names, top_n_display=10):
    """Display top optimization results summary for three NSGA-III runs."""
    print("\n--- Top 10 Optimal Results from Pareto Front for Three NSGA-III Runs ---")

    columns_to_show = ["optimizer", "total", "mechanical", "volumetric", "dK", "dZ", "LKG", "lF", "zeta"]

    for df, name in zip(df_list, folder_names):
        if not df.empty and "mechanical" in df.columns and "volumetric" in df.columns:
            # Filter valid data
            valid_data = df[(df["mechanical"] > 0) & (df["volumetric"] > 0) &
                            (df["mechanical"] < 1e6) & (df["volumetric"] < 1e6)]

            if not valid_data.empty:
                pareto_nsga3 = find_pareto_front(valid_data, ["mechanical", "volumetric"])

                if not pareto_nsga3.empty:
                    top_nsga3_results = pareto_nsga3.sort_values(by='total').head(top_n_display).copy()
                    top_nsga3_results["Optimizer Selection"] = "Pareto Front (Sorted by Total Loss)"

                    # Filter columns that exist
                    available_columns = [col for col in columns_to_show if col in top_nsga3_results.columns]
                    top_nsga3_results = top_nsga3_results[available_columns]

                    print(f"\n{name} Top {top_nsga3_results.shape[0]} (from Pareto Front, sorted by total loss):")
                    if not top_nsga3_results.empty:
                        for idx, (i, row) in enumerate(top_nsga3_results.iterrows()):
                            print(
                                f"{idx + 1}. Total Loss: {row['total']:.4e}, Mech Loss: {row.get('mechanical', 'N/A'):.4e}, Vol Loss: {row.get('volumetric', 'N/A'):.4e}")
                            print(
                                f"    Parameters: dK={row.get('dK', 'N/A'):.4f}, dZ={row.get('dZ', 'N/A'):.4f}, LKG={row.get('LKG', 'N/A'):.4f}, lF={row.get('lF', 'N/A'):.4f}, zeta={row.get('zeta', 'N/A'):.0f}")
                else:
                    print(f"  No Pareto optimal solutions found for {name}.")
            else:
                print(f"  No valid data for {name}.")
        else:
            print(f"  {name} data is empty or missing 'mechanical'/'volumetric' columns.")


def create_phase_labeled_plots_three_nsga(df_list, folder_names, output_dir):
    """Create plots specifically with Phase-I, Phase-II, Phase-III labels."""

    phase_labels = ["Phase-I", "Phase-II", "Phase-III"]

    # Create parameter evolution with phase labels
    # Prepare best iteration data for all three
    df_best_list = []
    for df in df_list:
        if not df.empty and 'iteration' in df.columns and 'total' in df.columns:
            try:
                best_indices = df.groupby('iteration')['total'].idxmin()
                df_best = df.loc[best_indices].copy()

                # Ensure clearance exists
                if 'clearance' not in df_best.columns and 'dZ' in df_best.columns and 'dK' in df_best.columns:
                    df_best['clearance'] = df_best['dZ'] - df_best['dK']

                df_best_list.append(df_best)
            except Exception as e:
                print(f"Error preparing phase-labeled data: {e}")
                df_best_list.append(pd.DataFrame())
        else:
            df_best_list.append(pd.DataFrame())

    parameters_with_clearance = ["clearance", "LKG", "lF", "zeta"]
    plots_to_show = parameters_with_clearance + ["convergence"]

    if any(not df.empty for df in df_best_list):
        fig, axes = plt.subplots(len(plots_to_show), 1, figsize=(12, 18), sharex=True)
        if len(plots_to_show) == 1:
            axes = [axes]

        colors = ['orange', 'green', 'purple']
        markers = ['x', 's', '^']
        linestyles = ['-', '--', '-.']

        for i, param in enumerate(plots_to_show):
            if param == "convergence":
                # Special handling for convergence plot
                for j, (df, phase_label) in enumerate(zip(df_list, phase_labels)):
                    if not df.empty and 'iteration' in df.columns and 'total' in df.columns:
                        try:
                            df_conv = safe_groupby_operation(df, 'iteration', 'total', 'min')
                            if not df_conv.empty:
                                axes[i].plot(df_conv['iteration'], df_conv['total'],
                                             marker=markers[j], linestyle=linestyles[j], color=colors[j],
                                             label=phase_label, linewidth=2, markersize=6)
                        except Exception as e:
                            print(f"Error plotting convergence for {phase_label}: {e}")

                axes[i].set_ylabel("Best Total Power Loss [W]", fontsize=14)
                axes[i].set_yscale('log')
            else:
                # Regular parameter plots
                for j, (df_best, phase_label) in enumerate(zip(df_best_list, phase_labels)):
                    if not df_best.empty and param in df_best.columns and 'iteration' in df_best.columns:
                        try:
                            axes[i].plot(df_best['iteration'], df_best[param],
                                         marker=markers[j], linestyle=linestyles[j], color=colors[j],
                                         label=phase_label, linewidth=2, markersize=6)
                        except Exception as e:
                            print(f"Error plotting {param} for {phase_label}: {e}")

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
            axes[i].legend(fontsize=12)

        axes[-1].set_xlabel("Generation", fontsize=14)
        axes[-1].tick_params(axis='both', labelsize=12)

        fig.suptitle("Parameter Evolution: Phase Comparison (Best per Generation)", fontsize=16, y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / "param_evolution_phase_labeled.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        phase_param_result = "param_evolution_phase_labeled.png"
    else:
        phase_param_result = "No data"

    print(f"Phase-labeled plots completed: {phase_param_result}")
    return [phase_param_result]


def main_parallel_three_nsga_complete_with_generation():
    """Enhanced main function supporting all three folder reading methods."""
    start_time = time.time()

    # Configuration for 3 NSGA-III folders - NOW SUPPORTS 3 METHODS
    folder_configs = [
        {
            'name': 'Generation-NSGA-III',
            'path': r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\run_zeta_5_Adaptive_selction_optimizer\genetic\parallel_runs',
            'method': 'generation'  # Uses original method
        },
        {
            'name': 'simple-NSGA-III',
            'path': r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run5_diameter_zeta_simple_nsga-III\simple_nsga3',
            'method': 'standard'  # Uses original method
        },
        {
            'name': 'Advance-NSGA-III',
            'path': r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run4_length_diameter_zeta_optimization_simple_nsga_more_generation\advanced_nsga3',
            'method': 'standard'  # Uses new generation method
        }
    ]

    pressure_threshold_mpa = 10.0
    max_penalty_percent = 200.0
    num_workers = min(mp.cpu_count(), 24)

    print("=" * 80)
    print("THREE NSGA-III ANALYSIS WITH MULTIPLE FOLDER READING METHODS")
    print("=" * 80)

    # Enhanced diagnostics
    print("\nRUNNING ENHANCED DIAGNOSTICS...")
    diagnose_folder_structure(folder_configs)

    # Load results with appropriate method for each folder
    print("\nLOADING OPTIMIZATION RESULTS WITH ADAPTIVE METHODS")
    print("=" * 60)
    load_start_time = time.time()

    results_list = []

    for config in folder_configs:
        print(f"\nProcessing {config['name']} using {config['method']} method...")

        try:
            if config['method'] == 'generation':
                # Use generation pattern loader
                results = load_results_generation_pattern_parallel(
                    config['path'],
                    pressure_threshold_mpa,
                    max_penalty_percent,
                    num_workers // 3
                )
            else:
                # Use standard loader
                results = load_results_with_contact_pressure_parallel(
                    config['path'],
                    "NSGA-III",
                    pressure_threshold_mpa,
                    max_penalty_percent,
                    num_workers // 3
                )

            # Ensure results is never None
            if results is None:
                results = []

        except Exception as e:
            print(f"Error loading results for {config['name']}: {e}")
            results = []

        results_list.append(results)
        print(f"Loaded {len(results)} results for {config['name']}")

    load_time = time.time() - load_start_time
    print(f"Data loading completed in {load_time:.2f} seconds")

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
            print(f"Processed {len(results)} valid results for {config['name']}")
        else:
            # Add empty DataFrame to maintain structure
            df_list.append(pd.DataFrame())
            folder_names.append(config['name'])
            print(f"No valid results for {config['name']}")

    # Validate DataFrame structures
    validate_dataframe_structure(df_list, folder_names)

    if all(len(df) == 0 for df in df_list):
        print("\nNO VALID RESULTS LOADED FROM ANY FOLDER!")
        print("   Please check the diagnostic output above to identify the issue.")
        return

    # Create output directory
    output_dir = Path("optimization_plots_three_nsga_complete")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print data summary
    print(f"\nDATA SUMMARY FOR THREE NSGA-III RUNS:")
    for i, (df, name) in enumerate(zip(df_list, folder_names)):
        if not df.empty:
            print(f"   {name} data shape: {df.shape}")
            if 'pattern_type' in df.columns:
                pattern_counts = df['pattern_type'].value_counts()
                for pattern, count in pattern_counts.items():
                    print(f"     - {pattern} format: {count} simulations")

            if 'contact_pressure_valid' in df.columns:
                valid_pressure = df[df['contact_pressure_valid'] == True]
                if not valid_pressure.empty and 'max_contact_pressure' in valid_pressure.columns:
                    print(f"   {name} Contact Pressure Statistics:")
                    print(f"     - Valid contact pressure data: {len(valid_pressure)}/{len(df)} simulations")
                    print(
                        f"     - Max contact pressure range: {valid_pressure['max_contact_pressure'].min():.1f} - {valid_pressure['max_contact_pressure'].max():.1f} MPa")
                    if 'contact_pressure_penalty' in valid_pressure.columns:
                        print(f"     - Average penalty: {valid_pressure['contact_pressure_penalty'].mean() * 100:.1f}%")
        else:
            print(f"   {name}: No valid data")

    # Create all plots in parallel
    print("\nCREATING COMPLETE PLOT SET FOR THREE NSGA-III COMPARISON")
    print("=" * 60)
    plot_start_time = time.time()

    # Create main comparison plots
    try:
        completed_plots = create_all_plots_parallel_complete_three_nsga(df_list, folder_names, output_dir,
                                                                        pressure_threshold_mpa, num_workers)
    except Exception as e:
        print(f"Error in main plotting: {e}")
        completed_plots = []

    # Create advanced analysis plots
    with ThreadPoolExecutor(max_workers=4) as executor:
        try:
            future_sensitivity = executor.submit(create_sensitivity_analysis_three_nsga_parallel,
                                                 df_list, folder_names, output_dir)
            future_comprehensive = executor.submit(create_comprehensive_comparison_plots_three_nsga,
                                                   df_list, folder_names, output_dir)
            future_phase_labeled = executor.submit(create_phase_labeled_plots_three_nsga,
                                                   df_list, folder_names, output_dir)
            future_statistical = executor.submit(create_statistical_comparison_three_nsga,
                                                 df_list, folder_names, output_dir)

            # Wait for completion
            sensitivity_result = future_sensitivity.result()
            comprehensive_result = future_comprehensive.result()
            phase_results = future_phase_labeled.result()
            statistical_result = future_statistical.result()

            print(f"  Sensitivity analysis: {sensitivity_result}")
            print(f"  Comprehensive comparison: {comprehensive_result}")
            print(f"  Phase-labeled plots: {phase_results}")
            print(f"  Statistical comparison: {statistical_result}")

        except Exception as e:
            print(f"Error in advanced plotting: {e}")

    plot_time = time.time() - plot_start_time
    print(f"Complete parallel plotting completed in {plot_time:.2f} seconds")

    # Save results
    print("\nSaving optimization results...")
    save_start_time = time.time()
    try:
        save_optimization_results_three_nsga_parallel(df_list, folder_names, output_dir, top_n_save=5)
    except Exception as e:
        print(f"Error saving results: {e}")
    save_time = time.time() - save_start_time

    # Print detailed best solution comparison
    print_detailed_best_solutions_three(df_list, folder_names)

    # Display top results summary
    print_top_results_summary_three_nsga(df_list, folder_names, top_n_display=10)

    total_time = time.time() - start_time

    # Print comprehensive performance summary
    print("\n" + "=" * 80)
    print("COMPLETE THREE NSGA-III PARALLEL OPTIMIZATION ANALYSIS FINISHED")
    print("=" * 80)
    print(f"Performance Summary:")
    print(f"  Data loading time: {load_time:.2f} seconds")
    print(f"  Plotting time: {plot_time:.2f} seconds")
    print(f"  Results saving time: {save_time:.2f} seconds")
    print(f"  Total execution time: {total_time:.2f} seconds")
    print(f"\nAll results saved to: {output_dir}")
    print(f"\n=== GENERATED PLOTS AND ANALYSIS ===")

    # Basic comparison plots
    print(f"📊 Basic Comparison Plots:")
    print(f"   - Three NSGA-III convergence comparison")
    print(f"   - Combined Pareto front comparison")
    print(f"   - Parameter evolution comparison with clearance")
    print(f"   - Contact pressure vs loss analysis for all three runs")
    print(f"   - Penalized vs non-penalized convergence comparison")
    print(f"   - All points parameter evolution with best solution markers (NEW!)")

    # Individual plots
    print(f"\n📋 Individual Run Analysis:")
    print(f"   - Individual parameter distributions for each run")
    print(f"   - Individual parameter vs loss scatter plots")
    print(f"   - Individual pairwise parameter relationship plots")
    print(f"   - Individual Pareto front plots")

    # Advanced analysis
    print(f"\n🔬 Advanced Analysis:")
    print(f"   - Parameter sensitivity comparison across all three runs")
    print(f"   - Comprehensive statistical comparison")
    print(f"   - Phase-labeled parameter evolution plots (Phase-I, Phase-II, Phase-III)")
    print(f"   - Statistical comparison with performance improvement metrics")
    print(f"   - All selected points visualization with jitter and best markers")

    # Generated files
    print(f"\n📁 Generated Files:")
    print(f"   - top_optimal_results_three_NSGA_with_contact_penalty.csv")
    print(f"   - Individual results files for each NSGA-III run")
    print(f"   - statistical_comparison_three_nsga.csv")
    print(f"   - Individual plot directories for each run")

    print("=" * 80)
    print("✅ COMPLETE three NSGA-III optimization analysis with robust error handling!")
    print("🎯 All plots now include Phase-I, Phase-II, Phase-III labeling as requested!")
    print("🔧 Fixed data structure issues and added proper geometry parameter extraction!")
    print("⭐ NEW: Added all points plot with best solution markers and jitter for clarity!")
    print("=" * 80)


if __name__ == "__main__":
    # Set start method for multiprocessing (important for Windows)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set

    main_parallel_three_nsga_complete_with_generation()