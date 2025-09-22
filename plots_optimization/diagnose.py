from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.stats import pearsonr
import os

# Import helper functions from the local T1 module; fall back to direct import
# when the script is executed standalone.
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

# --- Flexible Parameter Parsing Functions ---

# Define both regex patterns
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
            print(f"📁 New format detected: {folder_name}")

            if geometry_params["dK"] is not None:
                params["dK"] = geometry_params["dK"]

                # Calculate clearance from extracted dK and parsed dZ
                calculated_clearance = params["dZ"] - params["dK"]
                parsed_clearance = params["CL"]

                # Verify clearance values match (with tolerance)
                if abs(calculated_clearance - parsed_clearance) > 0.001:  # 1 micron tolerance
                    print(f"⚠️  Clearance mismatch in {folder_name}")
                    print(f"     Parsed CL: {parsed_clearance:.6f} mm")
                    print(f"     Calculated (dZ-dK): {calculated_clearance:.6f} mm")

                # Use the calculated clearance based on extracted dK
                params["clearance"] = calculated_clearance

            else:
                print(f"⚠️  Could not extract dK for {folder_name}, using parsed clearance")
                params["dK"] = params["dZ"] - params["CL"]  # Fallback calculation
                params["clearance"] = params["CL"]

        elif pattern_type == "old":
            # OLD FORMAT: Folder has dK, calculate clearance as dZ - dK
            print(f"📁 Old format detected: {folder_name}")

            # dK is already in params from folder name
            params["clearance"] = params["dZ"] - params["dK"]

            # If geometry file has dK, compare with folder value
            if geometry_params["dK"] is not None:
                geometry_dK = geometry_params["dK"]
                folder_dK = params["dK"]

                if abs(geometry_dK - folder_dK) > 0.001:
                    print(f"ℹ️  dK mismatch in {folder_name}")
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
                print(f"ℹ️  Using lF from geometry file for {folder_name}: {geometry_lF:.3f} mm")

            params["lF"] = geometry_lF

        print(f"✅ Parsed parameters: dK={params['dK']:.4f}, dZ={params['dZ']:.4f}, clearance={params['clearance']:.4f}")
        return params

    except ValueError as e:
        print(f"❌ Error parsing parameters from {folder_name}: {e}")
        return {}


def load_contact_pressure(sim_path: str) -> Dict[str, float]:
    """Load contact pressure data from Piston_Contact_Pressure.txt file"""
    contact_pressure_path = Path(sim_path) / "output" / "piston" / "matlab" / "Piston_Contact_Pressure.txt"

    if not contact_pressure_path.exists():
        print(f"⚠️  Contact pressure file not found: {contact_pressure_path}")
        return {"max_contact_pressure": 0.0, "avg_contact_pressure": 0.0, "contact_pressure_valid": False}

    try:
        # Load contact pressure data using the T1 function
        pressure_data = load_matlab_txt(str(contact_pressure_path))

        if pressure_data is None or pressure_data.size == 0:
            print(f"⚠️  Empty contact pressure data: {sim_path}")
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
        print(f"❌ Error loading contact pressure data from {sim_path}: {e}")
        return {"max_contact_pressure": 0.0, "avg_contact_pressure": 0.0, "contact_pressure_valid": False}


def calculate_contact_pressure_penalty(max_contact_pressure_mpa: float,
                                       pressure_threshold_mpa: float = 100.0,
                                       max_penalty_percent: float = 200.0) -> float:
    """
    Calculate contact pressure penalty based on maximum contact pressure.

    Args:
        max_contact_pressure_mpa: Maximum contact pressure in MPa
        pressure_threshold_mpa: Pressure threshold (10^7 Pa = 10 MPa by default, but using 100 MPa as more realistic)
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
        print(f"❌ Error parsing loss file {sim_path}: {e}")
        return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}


def load_results_with_contact_pressure(folder_path: str, opt_type: str,
                                       pressure_threshold_mpa: float = 100.0,
                                       max_penalty_percent: float = 200.0) -> list:
    """Updated load_results function with contact pressure penalty calculation"""
    base_folder = Path(folder_path)
    results = []
    if not base_folder.exists():
        print(f"❌ Base folder not found: {base_folder}")
        return []

    subfolders = [f for f in base_folder.iterdir() if f.is_dir()]
    print(f"📁 Found {len(subfolders)} subfolders in {base_folder}")

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
            print(f"❌ Unknown optimization type: {opt_type}")
            continue

        sim_folders = [f for f in folder.iterdir() if f.is_dir()]
        folder_results = 0

        for sim in sim_folders:
            # Use flexible pattern matching
            pattern_type, _ = detect_folder_pattern(sim.name)
            if pattern_type == "none":
                continue

            params = parse_parameters_flexible(sim.name, str(sim))
            if not params:
                continue

            # Load losses
            losses = parse_loss_file(str(sim))

            # Load contact pressure data
            contact_pressure_data = load_contact_pressure(str(sim))

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
            record["folder_name"] = str(sim)
            record["optimizer"] = opt_type
            record["pattern_type"] = pattern_type

            if record.get("valid") and record["total"] < 1e6:
                results.append(record)
                folder_results += 1

        if folder_results > 0:
            print(f"  ✅ {folder_name}: {folder_results} valid simulations")

    return results


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


def create_enhanced_combined_plots_with_penalty(df_bo, df_nsga3, output_dir,
                                                pressure_threshold_mpa=100.0):
    """Create enhanced combined plots that highlight the best solutions with contact pressure penalty."""

    # Find the best solutions using penalized total loss
    best_bo_idx = None
    best_nsga3_idx = None

    # Also find best solutions without penalty for comparison
    best_bo_no_penalty_idx = None
    best_nsga3_no_penalty_idx = None

    if not df_bo.empty:
        best_bo_idx = df_bo['penalized_total'].idxmin()
        best_bo_solution = df_bo.loc[best_bo_idx]

        best_bo_no_penalty_idx = df_bo['total'].idxmin()
        best_bo_no_penalty = df_bo.loc[best_bo_no_penalty_idx]

        print(f"🏆 Best BO solution (with penalty): Penalized Loss = {best_bo_solution['penalized_total']:.2e} W, "
              f"Contact Pressure = {best_bo_solution['max_contact_pressure']:.1f} MPa, "
              f"Penalty = {best_bo_solution['contact_pressure_penalty'] * 100:.1f}%")

        print(f"🔍 Best BO solution (without penalty): Total Loss = {best_bo_no_penalty['total']:.2e} W, "
              f"Contact Pressure = {best_bo_no_penalty['max_contact_pressure']:.1f} MPa")

    if not df_nsga3.empty:
        best_nsga3_idx = df_nsga3['penalized_total'].idxmin()
        best_nsga3_solution = df_nsga3.loc[best_nsga3_idx]

        best_nsga3_no_penalty_idx = df_nsga3['total'].idxmin()
        best_nsga3_no_penalty = df_nsga3.loc[best_nsga3_no_penalty_idx]

        print(
            f"🏆 Best NSGA-III solution (with penalty): Penalized Loss = {best_nsga3_solution['penalized_total']:.2e} W, "
            f"Contact Pressure = {best_nsga3_solution['max_contact_pressure']:.1f} MPa, "
            f"Penalty = {best_nsga3_solution['contact_pressure_penalty'] * 100:.1f}%")

        print(f"🔍 Best NSGA-III solution (without penalty): Total Loss = {best_nsga3_no_penalty['total']:.2e} W, "
              f"Contact Pressure = {best_nsga3_no_penalty['max_contact_pressure']:.1f} MPa")

    # 1. Enhanced Combined Convergence Plot with Penalty
    plt.figure(figsize=(12, 8))
    plot_data_exists = False

    if not df_bo.empty:
        # Regular convergence (without penalty)
        df_bo_best_conv = df_bo.groupby('iteration')['total'].min().reset_index()
        plt.subplot(2, 1, 1)
        plt.plot(df_bo_best_conv['iteration'], df_bo_best_conv['total'],
                 marker='o', linestyle='-', color='blue', label='BO (Total Loss)', linewidth=2, markersize=6)

        # Penalized convergence
        df_bo_best_conv_penalty = df_bo.groupby('iteration')['penalized_total'].min().reset_index()
        plt.plot(df_bo_best_conv_penalty['iteration'], df_bo_best_conv_penalty['penalized_total'],
                 marker='s', linestyle='--', color='darkblue', label='BO (Penalized)', linewidth=2, markersize=6)

        plot_data_exists = True

    if not df_nsga3.empty:
        if not plot_data_exists:
            plt.subplot(2, 1, 1)

        # Regular convergence (without penalty)
        df_nsga3_best_conv = df_nsga3.groupby('iteration')['total'].min().reset_index()
        plt.plot(df_nsga3_best_conv['iteration'], df_nsga3_best_conv['total'],
                 marker='x', linestyle='-', color='orange', label='NSGA-III (Total Loss)', linewidth=2, markersize=8)

        # Penalized convergence
        df_nsga3_best_conv_penalty = df_nsga3.groupby('iteration')['penalized_total'].min().reset_index()
        plt.plot(df_nsga3_best_conv_penalty['iteration'], df_nsga3_best_conv_penalty['penalized_total'],
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
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 2. Contact Pressure vs Iteration subplot
        plt.subplot(2, 1, 2)

        if not df_bo.empty:
            df_bo_pressure = df_bo.groupby('iteration')['max_contact_pressure'].min().reset_index()
            plt.plot(df_bo_pressure['iteration'], df_bo_pressure['max_contact_pressure'],
                     marker='o', linestyle='-', color='blue', label='BO Min Pressure', linewidth=2, markersize=6)

        if not df_nsga3.empty:
            df_nsga3_pressure = df_nsga3.groupby('iteration')['max_contact_pressure'].min().reset_index()
            plt.plot(df_nsga3_pressure['iteration'], df_nsga3_pressure['max_contact_pressure'],
                     marker='x', linestyle='-', color='orange', label='NSGA-III Min Pressure', linewidth=2,
                     markersize=8)

        # Add threshold line
        if plot_data_exists:
            plt.axhline(y=pressure_threshold_mpa, color='red', linestyle=':',
                        label=f'Pressure Threshold ({pressure_threshold_mpa} MPa)', linewidth=2)

        plt.xlabel("Optimization Step (Iteration/Generation)", fontsize=14)
        plt.ylabel("Max Contact Pressure [MPa]", fontsize=14)
        plt.title("Contact Pressure Evolution", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / "enhanced_convergence_with_penalty.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Enhanced convergence plot with penalty saved")

    # 3. Contact Pressure vs Total Loss Scatter Plot
    if plot_data_exists:
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
            if best_bo_idx is not None:
                plt.scatter([best_bo_solution["total"]], [best_bo_solution["max_contact_pressure"]],
                            color='red', s=200, marker='*', edgecolors='black', linewidth=2,
                            label=f'Best BO (with penalty)', zorder=5)

            if best_bo_no_penalty_idx is not None and best_bo_no_penalty_idx != best_bo_idx:
                plt.scatter([best_bo_no_penalty["total"]], [best_bo_no_penalty["max_contact_pressure"]],
                            color='lightcoral', s=150, marker='o', edgecolors='black', linewidth=2,
                            label=f'Best BO (no penalty)', zorder=4)

        if not df_nsga3.empty:
            plt.scatter(df_nsga3["total"], df_nsga3["max_contact_pressure"],
                        color='orange', alpha=0.6, s=30, label='NSGA-III')

            # Highlight best solutions
            if best_nsga3_idx is not None:
                plt.scatter([best_nsga3_solution["total"]], [best_nsga3_solution["max_contact_pressure"]],
                            color='darkred', s=200, marker='*', edgecolors='black', linewidth=2,
                            label=f'Best NSGA-III (with penalty)', zorder=5)

            if best_nsga3_no_penalty_idx is not None and best_nsga3_no_penalty_idx != best_nsga3_idx:
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
        print("✅ Contact pressure analysis plot saved")


def create_all_additional_plots(df_bo, df_nsga3, output_dir):
    """Create all the additional plots from the second code"""

    # Define parameter names for iteration
    parameters = ["dK", "dZ", "LKG", "lF", "zeta"]

    # Ensure a common iteration column name for both
    if 'generation' in df_nsga3.columns:
        df_nsga3 = df_nsga3.rename(columns={'generation': 'iteration'})

    # 1. Individual Convergence Plots
    # BO Convergence Plot
    plt.figure(figsize=(8, 5))
    if not df_bo.empty:
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

    # NSGA-III Convergence Plot
    plt.figure(figsize=(8, 5))
    if not df_nsga3.empty:
        df_nsga3_best = df_nsga3.groupby('iteration')['total'].min().reset_index()
        plt.plot(df_nsga3_best['iteration'], df_nsga3_best['total'], color='orange', marker='o', label='NSGA-III')
    plt.xlabel("Generation")
    plt.ylabel("Best Total Loss [W]")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "convergence_NSGA3.png")
    plt.close()

    # 2. Combined Parameter Evolution Plot with Clearance and Convergence
    df_bo_best_iter_combined = pd.DataFrame()
    if not df_bo.empty:
        df_bo_best_iter_combined = df_bo.loc[df_bo.groupby('iteration')['total'].idxmin()].copy()
        # Calculate clearance for BO
        df_bo_best_iter_combined['clearance'] = df_bo_best_iter_combined['dZ'] - df_bo_best_iter_combined['dK']

    df_nsga3_best_iter_combined = pd.DataFrame()
    if not df_nsga3.empty:
        df_nsga3_best_iter_combined = df_nsga3.loc[df_nsga3.groupby('iteration')['total'].idxmin()].copy()
        # Calculate clearance for NSGA-III
        df_nsga3_best_iter_combined['clearance'] = df_nsga3_best_iter_combined['dZ'] - df_nsga3_best_iter_combined['dK']

    # Update parameters list to include clearance instead of dK and dZ, plus convergence
    parameters_with_clearance = ["clearance", "LKG", "lF", "zeta"]

    # Add convergence plot to the plots (total power loss evolution)
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

                # Calculate y-limits for parameters
                all_param_values = []
                if not df_bo_best_iter_combined.empty and param in df_bo_best_iter_combined.columns:
                    all_param_values.extend(df_bo_best_iter_combined[param].tolist())
                if not df_nsga3_best_iter_combined.empty and param in df_nsga3_best_iter_combined.columns:
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
        fig.savefig(output_dir / "combined_param_evolution_with_clearance_and_convergence.png", dpi=300)
        plt.close(fig)

    # 3. Parameter Distribution Histograms for BO
    if not df_bo.empty:
        fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
        for i, param in enumerate(parameters):
            sns.histplot(df_bo, x=param, kde=True, ax=axes[i], color='blue', bins=20)
            axes[i].set_title(f"BO: Distribution of {param}")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_dir / "param_distribution_BO.png")
        plt.close(fig)

    # 4. Parameter Distribution Histograms for NSGA-III
    if not df_nsga3.empty:
        fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
        for i, param in enumerate(parameters):
            sns.histplot(df_nsga3, x=param, kde=True, ax=axes[i], color='orange', bins=20)
            axes[i].set_title(f"NSGA-III: Distribution of {param}")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_dir / "param_distribution_NSGA3.png")
        plt.close(fig)

    # 5. Parameter vs Total Loss Scatter Plots for BO
    if not df_bo.empty:
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

    # 6. Parameter vs Total Loss Scatter Plots for NSGA-III
    if not df_nsga3.empty:
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

    # 7. Pair Plot for BO (parameters vs parameters and vs total)
    if not df_bo.empty:
        cols_for_pairplot = parameters + ["total"]
        # Use a sample if there are too many points for clarity
        df_plot = df_bo.copy()
        if len(df_plot) > 5000:
            df_plot = df_plot.sample(n=5000, random_state=42)
        sns.pairplot(df_plot[cols_for_pairplot], diag_kind="kde")
        plt.suptitle("BO Pairwise Parameter Relationships", y=1.02)
        plt.savefig(output_dir / "pairplot_BO.png")
        plt.close()

    # 8. Pair Plot for NSGA-III
    if not df_nsga3.empty:
        cols_for_pairplot = parameters + ["total"]
        df_plot = df_nsga3.copy()
        if len(df_plot) > 5000:
            df_plot = df_plot.sample(n=5000, random_state=42)
        sns.pairplot(df_plot[cols_for_pairplot], diag_kind="kde")
        plt.suptitle("NSGA-III Pairwise Parameter Relationships", y=1.02)
        plt.savefig(output_dir / "pairplot_NSGA3.png")
        plt.close()

    # 9. Pareto Front Scatter (Mechanical vs Volumetric Loss) for BO
    if not df_bo.empty and "mechanical" in df_bo.columns and "volumetric" in df_bo.columns:
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

    # 10. Pareto Front Scatter for NSGA-III
    if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
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

    # 11. Combined Parameter Evolution Plot (Original Style)
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

    # 12. Combined Convergence Plot
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

    # 13. Combined Pareto Front Plot
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

    # 14. Sensitivity Bar Chart
    # Compute clearance
    df_bo["clearance"] = df_bo["dZ"] - df_bo["dK"]
    df_nsga3["clearance"] = df_nsga3["dZ"] - df_nsga3["dK"]

    # Parameters to analyze
    selected_params = ["clearance", "zeta", "lF", "LKG"]
    # Human-readable labels
    param_labels = {
        "clearance": "clearance",
        "zeta": "γ",
        "lF": "lF",
        "LKG": "LKG"
    }

    sensitivity_data = []

    for param in selected_params:
        bo_corr = np.nan
        nsga_corr = np.nan

        if not df_bo.empty and param in df_bo.columns:
            bo_corr, _ = pearsonr(df_bo[param], df_bo["total"])
            bo_corr = abs(bo_corr)

        if not df_nsga3.empty and param in df_nsga3.columns:
            nsga_corr, _ = pearsonr(df_nsga3[param], df_nsga3["total"])
            nsga_corr = abs(nsga_corr)

        sensitivity_data.append({
            "parameter": param_labels[param],
            "BO": bo_corr,
            "NSGA-III": nsga_corr
        })

    # Convert to DataFrame
    df_sensitivity = pd.DataFrame(sensitivity_data)

    # Plotting
    x = np.arange(len(df_sensitivity["parameter"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, df_sensitivity["BO"], width, label='BO', color='blue')
    bars2 = ax.bar(x + width / 2, df_sensitivity["NSGA-III"], width, label='NSGA-III', color='orange')

    ax.set_xlabel("Parameter", fontsize=16)
    ax.set_ylabel("Sensitivity (|Pearson Correlation|)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sensitivity["parameter"], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=18)
    ax.grid(True, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_bar_chart_all_params.png")
    plt.close(fig)

    print("✅ All additional plots saved")


def create_penalty_distribution_analysis(df_bo, df_nsga3, output_dir, pressure_threshold_mpa=100.0):
    """Create penalty distribution analysis plots"""
    if not df_bo.empty or not df_nsga3.empty:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        if not df_bo.empty:
            valid_bo = df_bo[df_bo['contact_pressure_valid'] == True]
            if not valid_bo.empty:
                plt.hist(valid_bo['contact_pressure_penalty'] * 100, bins=20, alpha=0.7, color='blue', label='BO')
        if not df_nsga3.empty:
            valid_nsga3 = df_nsga3[df_nsga3['contact_pressure_valid'] == True]
            if not valid_nsga3.empty:
                plt.hist(valid_nsga3['contact_pressure_penalty'] * 100, bins=20, alpha=0.7, color='orange',
                         label='NSGA-III')
        plt.xlabel('Penalty (%)')
        plt.ylabel('Frequency')
        plt.title('Contact Pressure Penalty Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        if not df_bo.empty:
            valid_bo = df_bo[df_bo['contact_pressure_valid'] == True]
            if not valid_bo.empty:
                plt.hist(valid_bo['max_contact_pressure'], bins=20, alpha=0.7, color='blue', label='BO')
        if not df_nsga3.empty:
            valid_nsga3 = df_nsga3[df_nsga3['contact_pressure_valid'] == True]
            if not valid_nsga3.empty:
                plt.hist(valid_nsga3['max_contact_pressure'], bins=20, alpha=0.7, color='orange', label='NSGA-III')
        plt.axvline(x=pressure_threshold_mpa, color='red', linestyle='--',
                    label=f'Threshold ({pressure_threshold_mpa} MPa)')
        plt.xlabel('Max Contact Pressure (MPa)')
        plt.ylabel('Frequency')
        plt.title('Max Contact Pressure Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        if not df_bo.empty:
            plt.scatter(df_bo['total'], df_bo['penalized_total'], color='blue', alpha=0.6, s=30, label='BO')
        if not df_nsga3.empty:
            plt.scatter(df_nsga3['total'], df_nsga3['penalized_total'], color='orange', alpha=0.6, s=30,
                        label='NSGA-III')

        # Add diagonal line (no penalty line)
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
        print("✅ Penalty distribution analysis saved")


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
            print("❌ FOLDER DOES NOT EXIST!")
            continue

        # Get all subfolders
        base_folder = Path(folder_path)
        subfolders = [f for f in base_folder.iterdir() if f.is_dir()]
        print(f"✅ Found {len(subfolders)} subfolders")

        # Check first few subfolders to understand structure
        print(f"\n📁 First 10 subfolders:")
        for i, folder in enumerate(subfolders[:10]):
            folder_name = folder.name
            print(f"  {i + 1:2d}. {folder_name}")

            # Check if this looks like iteration/generation folder
            if optimizer_name == "BO":
                if folder_name.startswith("Iteration_I") or folder_name == "Initial_Sampling":
                    print(f"      ✅ Valid {optimizer_name} iteration folder")
                else:
                    print(f"      ❌ Not a recognized {optimizer_name} iteration folder")
            else:  # NSGA-III
                if folder_name.startswith("Generation_G") or folder_name == "Initial_Sampling":
                    print(f"      ✅ Valid {optimizer_name} generation folder")
                else:
                    print(f"      ❌ Not a recognized {optimizer_name} generation folder")

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
            print(f"\n🔍 Examining simulation folders in: {valid_iteration_folder.name}")
            sim_folders = [f for f in valid_iteration_folder.iterdir() if f.is_dir()]
            print(f"   Found {len(sim_folders)} simulation folders")

            print(f"\n📋 Pattern analysis for first 10 simulation folders:")
            new_pattern_matches = 0
            old_pattern_matches = 0

            for i, sim in enumerate(sim_folders[:10]):
                sim_name = sim.name
                print(f"     {i + 1:2d}. {sim_name}")

                # Check both patterns
                pattern_type, match = detect_folder_pattern(sim_name)

                if pattern_type == "new":
                    print(f"        ✅ MATCHES NEW pattern (with CL)")
                    new_pattern_matches += 1
                    params = {k: float(v) for k, v in match.groupdict().items()}
                    print(
                        f"        📊 Parsed: CL={params['CL']}, dZ={params['dZ']}, LKG={params['LKG']}, lF={params['lF']}, zeta={params['zeta']}")
                elif pattern_type == "old":
                    print(f"        ✅ MATCHES OLD pattern (with dK)")
                    old_pattern_matches += 1
                    params = {k: float(v) for k, v in match.groupdict().items()}
                    print(
                        f"        📊 Parsed: dK={params['dK']}, dZ={params['dZ']}, LKG={params['LKG']}, lF={params['lF']}, zeta={params['zeta']}")
                else:
                    print(f"        ❌ Does NOT match either pattern")

            print(f"\n📈 Pattern matching summary for {valid_iteration_folder.name}:")
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
                    print(f"\n📄 Checking file structure in: {matching_folder.name}")

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
                            f"     {'✅' if exists else '❌'} {file_path}: {'exists' if exists else 'missing'} {f'({size} bytes)' if exists else ''}")
        else:
            print(f"\n❌ No valid iteration/generation folders found for {optimizer_name}")


def main():
    """Main function with contact pressure penalty-based optimization analysis."""

    # Update these paths to your actual folders
    bo_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\bayesian_optimization'
    nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\advanced_nsga3'

    # Contact pressure penalty settings
    pressure_threshold_mpa = 100.0  # 10^7 Pa = 10 MPa, but using 100 MPa as more realistic threshold
    max_penalty_percent = 200.0  # 200% penalty at threshold

    print("=" * 80)
    print("OPTIMIZATION ANALYSIS WITH CONTACT PRESSURE PENALTY AND ALL PLOTS")
    print("=" * 80)
    print("📁 Supported formats:")
    print("   - NEW: CL0.028_dZ19.888_LKG56.4_lF36.9_zeta5 (extracts dK from geometry.txt)")
    print("   - OLD: dK19.48_dZ19.51_LKG56.4_lF36.9_zeta5 (calculates clearance as dZ-dK)")
    print(f"🎯 Contact Pressure Penalty Settings:")
    print(f"   - Pressure Threshold: {pressure_threshold_mpa:.1f} MPa")
    print(f"   - Maximum Penalty: {max_penalty_percent:.0f}%")
    print(f"   - Formula: penalized_loss = total_loss × (1 + penalty)")
    print(
        f"   - Penalty = min({max_penalty_percent:.0f}%, (max_pressure/{pressure_threshold_mpa:.0f}) × {max_penalty_percent:.0f}%)")
    print("=" * 80)

    # Run diagnostics first
    print("\n🔍 RUNNING DIAGNOSTICS...")
    diagnose_folder_structure(bo_folder, nsga3_folder)

    # Load results using flexible parsing with contact pressure penalty
    print("\n" + "=" * 60)
    print("LOADING OPTIMIZATION RESULTS WITH CONTACT PRESSURE ANALYSIS")
    print("=" * 60)

    bo_results = load_results_with_contact_pressure(bo_folder, "BO", pressure_threshold_mpa, max_penalty_percent)
    print(f"✅ Loaded {len(bo_results)} valid BO results")

    nsga3_results = load_results_with_contact_pressure(nsga3_folder, "NSGA-III", pressure_threshold_mpa,
                                                       max_penalty_percent)
    print(f"✅ Loaded {len(nsga3_results)} valid NSGA-III results")

    if len(bo_results) == 0 and len(nsga3_results) == 0:
        print("\n❌ NO VALID RESULTS LOADED!")
        print("   Please check the diagnostic output above to identify the issue.")
        print("   Common issues:")
        print("   1. Folder names don't match expected patterns")
        print("   2. Required files (piston.txt, Piston_Contact_Pressure.txt) are missing")
        print("   3. Simulation results are invalid (total loss >= 1e6)")
        return

    # Convert to DataFrame for convenience
    df_bo = pd.DataFrame(bo_results)
    df_nsga3 = pd.DataFrame(nsga3_results)

    # Ensure a common iteration column name for both
    if 'generation' in df_nsga3.columns:
        df_nsga3 = df_nsga3.rename(columns={'generation': 'iteration'})

    # Create output directory for plots
    output_dir = Path("optimization_plots_with_contact_penalty_and_all_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 DATA SUMMARY WITH CONTACT PRESSURE ANALYSIS:")
    if not df_bo.empty:
        print(f"   BO data shape: {df_bo.shape}")
        pattern_counts_bo = df_bo['pattern_type'].value_counts()
        for pattern, count in pattern_counts_bo.items():
            print(f"     - {pattern} format: {count} simulations")

        # Contact pressure statistics
        valid_pressure_bo = df_bo[df_bo['contact_pressure_valid'] == True]
        if not valid_pressure_bo.empty:
            print(f"   BO Contact Pressure Statistics:")
            print(f"     - Valid contact pressure data: {len(valid_pressure_bo)}/{len(df_bo)} simulations")
            print(
                f"     - Max contact pressure range: {valid_pressure_bo['max_contact_pressure'].min():.1f} - {valid_pressure_bo['max_contact_pressure'].max():.1f} MPa")
            print(f"     - Average penalty: {valid_pressure_bo['contact_pressure_penalty'].mean() * 100:.1f}%")
            print(
                f"     - Simulations exceeding threshold: {(valid_pressure_bo['max_contact_pressure'] > pressure_threshold_mpa).sum()}")

    if not df_nsga3.empty:
        print(f"   NSGA-III data shape: {df_nsga3.shape}")
        pattern_counts_nsga3 = df_nsga3['pattern_type'].value_counts()
        for pattern, count in pattern_counts_nsga3.items():
            print(f"     - {pattern} format: {count} simulations")

        # Contact pressure statistics
        valid_pressure_nsga3 = df_nsga3[df_nsga3['contact_pressure_valid'] == True]
        if not valid_pressure_nsga3.empty:
            print(f"   NSGA-III Contact Pressure Statistics:")
            print(f"     - Valid contact pressure data: {len(valid_pressure_nsga3)}/{len(df_nsga3)} simulations")
            print(
                f"     - Max contact pressure range: {valid_pressure_nsga3['max_contact_pressure'].min():.1f} - {valid_pressure_nsga3['max_contact_pressure'].max():.1f} MPa")
            print(f"     - Average penalty: {valid_pressure_nsga3['contact_pressure_penalty'].mean() * 100:.1f}%")
            print(
                f"     - Simulations exceeding threshold: {(valid_pressure_nsga3['max_contact_pressure'] > pressure_threshold_mpa).sum()}")

    # Create enhanced plots with contact pressure penalty analysis
    print("\n" + "=" * 60)
    print("CREATING ENHANCED PLOTS WITH CONTACT PRESSURE PENALTY ANALYSIS")
    print("=" * 60)

    create_enhanced_combined_plots_with_penalty(df_bo, df_nsga3, output_dir, pressure_threshold_mpa)

    # Create all additional plots from the second code
    print("\n" + "=" * 60)
    print("CREATING ALL ADDITIONAL PLOTS")
    print("=" * 60)

    create_all_additional_plots(df_bo, df_nsga3, output_dir)

    # Create penalty distribution analysis
    create_penalty_distribution_analysis(df_bo, df_nsga3, output_dir, pressure_threshold_mpa)

    # Save Top 5 Optimal Results (by Total Loss) for Each Optimizer
    top_n_save = 5
    columns_to_save = ["optimizer", "iteration", "total", "penalized_total", "mechanical", "volumetric",
                       "max_contact_pressure", "avg_contact_pressure", "contact_pressure_penalty",
                       "dK", "dZ", "clearance", "LKG", "lF", "zeta", "pattern_type", "contact_pressure_valid"]

    # Get top results from each optimizer (based on penalized total)
    top_bo_penalty = pd.DataFrame()
    if not df_bo.empty:
        top_bo_penalty = df_bo.nsmallest(top_n_save, 'penalized_total').copy()
        top_bo_penalty["optimizer"] = "BO"
        if 'clearance' not in top_bo_penalty.columns:
            top_bo_penalty['clearance'] = top_bo_penalty['dZ'] - top_bo_penalty['dK']
        top_bo_penalty = top_bo_penalty[columns_to_save]

    top_nsga3_penalty = pd.DataFrame()
    if not df_nsga3.empty:
        top_nsga3_penalty = df_nsga3.nsmallest(top_n_save, 'penalized_total').copy()
        top_nsga3_penalty["optimizer"] = "NSGA-III"
        if 'clearance' not in top_nsga3_penalty.columns:
            top_nsga3_penalty['clearance'] = top_nsga3_penalty['dZ'] - top_nsga3_penalty['dK']
        top_nsga3_penalty = top_nsga3_penalty[columns_to_save]

    # Also get top results without penalty for comparison
    top_bo_no_penalty = pd.DataFrame()
    if not df_bo.empty:
        top_bo_no_penalty = df_bo.nsmallest(top_n_save, 'total').copy()
        top_bo_no_penalty["optimizer"] = "BO_no_penalty"
        if 'clearance' not in top_bo_no_penalty.columns:
            top_bo_no_penalty['clearance'] = top_bo_no_penalty['dZ'] - top_bo_no_penalty['dK']
        top_bo_no_penalty = top_bo_no_penalty[columns_to_save]

    top_nsga3_no_penalty = pd.DataFrame()
    if not df_nsga3.empty:
        top_nsga3_no_penalty = df_nsga3.nsmallest(top_n_save, 'total').copy()
        top_nsga3_no_penalty["optimizer"] = "NSGA-III_no_penalty"
        if 'clearance' not in top_nsga3_no_penalty.columns:
            top_nsga3_no_penalty['clearance'] = top_nsga3_no_penalty['dZ'] - top_nsga3_no_penalty['dK']
        top_nsga3_no_penalty = top_nsga3_no_penalty[columns_to_save]

    # Combine and save
    df_top_combined = pd.concat([top_bo_penalty, top_nsga3_penalty,
                                 top_bo_no_penalty, top_nsga3_no_penalty], ignore_index=True)
    df_top_combined.to_csv(output_dir / "top_optimal_results_with_contact_penalty.csv", index=False)

    # Also save the traditional format from the second code
    columns_to_save_traditional = ["optimizer", "iteration", "total", "mechanical", "volumetric", "dK", "dZ", "LKG",
                                   "lF", "zeta"]

    # Get top 5 BO results (traditional)
    top_bo_save_traditional = pd.DataFrame()
    if not df_bo.empty:
        top_bo_save_traditional = df_bo.nsmallest(top_n_save, 'total').copy()
        top_bo_save_traditional["optimizer"] = "BO"
        top_bo_save_traditional = top_bo_save_traditional[columns_to_save_traditional]

    # Get top 5 NSGA-III results (traditional)
    top_nsga3_save_traditional = pd.DataFrame()
    if not df_nsga3.empty:
        top_nsga3_save_traditional = df_nsga3.nsmallest(top_n_save, 'total').copy()
        top_nsga3_save_traditional["optimizer"] = "NSGA-III"
        top_nsga3_save_traditional = top_nsga3_save_traditional[columns_to_save_traditional]

    # Combine and save traditional format
    df_top_combined_traditional = pd.concat([top_bo_save_traditional, top_nsga3_save_traditional], ignore_index=True)
    df_top_combined_traditional.to_csv(output_dir / "top5_optimal_results.csv", index=False)

    # Print detailed best solution comparison
    print(f"\n🏆 BEST SOLUTIONS COMPARISON (WITH vs WITHOUT PENALTY):")
    print("=" * 80)

    if not df_bo.empty:
        # Best with penalty
        best_bo_penalty_idx = df_bo['penalized_total'].idxmin()
        best_bo_penalty = df_bo.loc[best_bo_penalty_idx]

        # Best without penalty
        best_bo_no_penalty_idx = df_bo['total'].idxmin()
        best_bo_no_penalty = df_bo.loc[best_bo_no_penalty_idx]

        print(f"\n🔵 BO BEST SOLUTIONS:")
        print(f"   📊 WITH PENALTY (Iteration {best_bo_penalty.get('iteration', 'N/A')}):")
        print(f"      - Total Loss: {best_bo_penalty['total']:.4e} W")
        print(f"      - Penalized Total: {best_bo_penalty['penalized_total']:.4e} W")
        print(f"      - Max Contact Pressure: {best_bo_penalty['max_contact_pressure']:.1f} MPa")
        print(f"      - Penalty: {best_bo_penalty['contact_pressure_penalty'] * 100:.1f}%")
        print(
            f"      - Parameters: dK={best_bo_penalty.get('dK', 0):.3f}, dZ={best_bo_penalty.get('dZ', 0):.3f}, clearance={best_bo_penalty.get('clearance', 0):.3f}")

        print(f"   📊 WITHOUT PENALTY (Iteration {best_bo_no_penalty.get('iteration', 'N/A')}):")
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

        print(f"\n🟠 NSGA-III BEST SOLUTIONS:")
        print(f"   📊 WITH PENALTY (Generation {best_nsga3_penalty.get('iteration', 'N/A')}):")
        print(f"      - Total Loss: {best_nsga3_penalty['total']:.4e} W")
        print(f"      - Penalized Total: {best_nsga3_penalty['penalized_total']:.4e} W")
        print(f"      - Max Contact Pressure: {best_nsga3_penalty['max_contact_pressure']:.1f} MPa")
        print(f"      - Penalty: {best_nsga3_penalty['contact_pressure_penalty'] * 100:.1f}%")
        print(
            f"      - Parameters: dK={best_nsga3_penalty.get('dK', 0):.3f}, dZ={best_nsga3_penalty.get('dZ', 0):.3f}, clearance={best_nsga3_penalty.get('clearance', 0):.3f}")

        print(f"   📊 WITHOUT PENALTY (Generation {best_nsga3_no_penalty.get('iteration', 'N/A')}):")
        print(f"      - Total Loss: {best_nsga3_no_penalty['total']:.4e} W")
        print(f"      - Penalized Total: {best_nsga3_no_penalty['penalized_total']:.4e} W")
        print(f"      - Max Contact Pressure: {best_nsga3_no_penalty['max_contact_pressure']:.1f} MPa")
        print(f"      - Penalty: {best_nsga3_no_penalty['contact_pressure_penalty'] * 100:.1f}%")
        print(
            f"      - Parameters: dK={best_nsga3_no_penalty.get('dK', 0):.3f}, dZ={best_nsga3_no_penalty.get('dZ', 0):.3f}, clearance={best_nsga3_no_penalty.get('clearance', 0):.3f}")

    # Display top 10 results summary (similar to second code)
    print("\n--- Top 10 Optimal Results from Pareto Front (NSGA-III) and Lowest Total Loss (BO) ---")

    top_n_display = 10
    columns_to_show = ["optimizer", "total", "mechanical", "volumetric", "dK", "dZ", "LKG", "lF", "zeta", "folder_name"]

    # For BO: Get top N by total loss
    top_bo_results = pd.DataFrame()
    if not df_bo.empty:
        top_bo_results = df_bo.nsmallest(top_n_display, 'total').copy()
        top_bo_results["Optimizer Selection"] = "Lowest Total Loss"
        top_bo_results = top_bo_results[columns_to_show]
        print("\nBO Top {} (by lowest total loss):".format(len(top_bo_results)))
        if not top_bo_results.empty:
            for i, row in top_bo_results.iterrows():
                print(
                    f"{i + 1}. Total Loss: {row['total']:.4e}, Mech Loss: {row['mechanical']:.4e}, Vol Loss: {row['volumetric']:.4e}")
                print(
                    f"    Parameters: dK={row['dK']:.4f}, dZ={row['dZ']:.4f}, LKG={row['LKG']:.4f}, lF={row['lF']:.4f}, zeta={row['zeta']:.0f}")
        else:
            print("  No valid BO results to display.")

    # For NSGA-III: Find Pareto front and then select top N by total loss from Pareto front
    top_nsga3_results = pd.DataFrame()
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

    print("\n" + "=" * 80)
    print("ENHANCED OPTIMIZATION ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"📁 All results saved to: {output_dir}")
    print(f"📊 Generated plots:")
    print(f"   - Enhanced convergence plots with penalty comparison")
    print(f"   - Contact pressure vs loss analysis")
    print(f"   - Penalty distribution analysis")
    print(f"   - Individual convergence plots (BO and NSGA-III)")
    print(f"   - Combined parameter evolution with clearance and convergence")
    print(f"   - Parameter distribution histograms")
    print(f"   - Parameter vs total loss scatter plots")
    print(f"   - Pairwise parameter relationship plots")
    print(f"   - Pareto front plots")
    print(f"   - Combined plots (convergence, pareto, parameter evolution)")
    print(f"   - Sensitivity analysis bar chart")
    print(f"📄 Generated files:")
    print(f"   - top_optimal_results_with_contact_penalty.csv")
    print(f"   - top5_optimal_results.csv")
    print("=" * 80)
    print("✅ Complete optimization analysis with contact pressure penalty and all plots completed!")
    print(f"🎯 Key Features:")
    print(f"   - Contact pressure penalty-based optimization")
    print(f"   - Flexible folder naming pattern support (OLD and NEW formats)")
    print(f"   - Comprehensive plotting suite from both analysis approaches")
    print(f"   - Enhanced diagnostics and error handling")


if __name__ == "__main__":
    main()