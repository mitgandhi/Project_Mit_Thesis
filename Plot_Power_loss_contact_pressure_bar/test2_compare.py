# Part 1: Imports and Utility Functions
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

# Ensure matplotlib backend is set
import matplotlib

matplotlib.use('TkAgg')

# Import all functions from T1 for contact pressure analysis
sys.path.append('..')  # Add current directory to path
try:
    from plots_optimization.T1 import (
        load_matlab_txt, round2, parse_geometry, parse_operating_conditions,
        create_cumulative_pressure_map, collect_simulation_data,
        create_comparison_plots, create_overall_summary, run_contact_pressure_analysis
    )
except ImportError:
    print("Warning: T1 module not found. Contact pressure analysis will use fallback methods.")

# Update global styling
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.figsize': (10, 6),
    'font.family': 'serif'
})


def parse_folder_name(folder_name):
    """
    Parse folder name like 'V60N_S4_dp_350_n_2000_d_100' to extract parameters
    Returns: (pump_series, series_num, pressure, speed, displacement)
    """
    pattern = r'(\w+)_S(\d+)_dp_(\d+)_n_(\d+)_d_(\d+)'
    match = re.match(pattern, folder_name)
    if match:
        pump_series = match.group(1)
        series_num = int(match.group(2))
        pressure = int(match.group(3))
        speed = int(match.group(4))
        displacement = int(match.group(5))
        return pump_series, series_num, pressure, speed, displacement
    return None


def read_data_file(filename):
    """Read data file with multiple delimiter attempts"""
    try:
        # First try to read as standard CSV
        df = pd.read_csv(filename)
        # If that fails, try reading with different delimiters
        if len(df.columns) == 1:
            for delimiter in ['\t', ' ', ';', '|']:
                try:
                    df = pd.read_csv(filename, delimiter=delimiter)
                    if len(df.columns) > 1:
                        break
                except:
                    continue
        # Clean the data
        df = df.dropna(axis=1, how='all')  # Remove empty columns
        df = df.dropna(how='all')  # Remove empty rows
        return df
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        return None


def read_piston_data(file_path):
    """
    Read piston.txt file and return mean values of power losses
    """
    try:
        # Use the robust file reading function
        df = read_data_file(file_path)

        if df is None:
            return None, None

        # Check if required columns exist
        required_cols = ['Total_Mechanical_Power_Loss', 'Total_Volumetric_Power_Loss']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Missing columns in {file_path}: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None, None

        # Calculate means for the required columns
        mech_power_loss_mean = df['Total_Mechanical_Power_Loss'].mean()
        vol_power_loss_mean = df['Total_Volumetric_Power_Loss'].mean()

        return mech_power_loss_mean, vol_power_loss_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


def calculate_total_force(mechanical_loss, volumetric_loss, speed_rpm):
    """
    Calculate total force from power losses
    Force approximation: F ≈ Power / (angular_velocity * effective_radius)
    This is a simplified calculation - adjust based on your specific methodology
    """
    total_power = mechanical_loss + volumetric_loss  # in Watts
    angular_velocity = speed_rpm * 2 * np.pi / 60  # convert RPM to rad/s

    # Assuming an effective radius (this should be based on your pump geometry)
    # You may need to adjust this based on your specific pump dimensions
    effective_radius = 0.05  # 50 mm in meters - ADJUST THIS VALUE

    total_force = total_power / (angular_velocity * effective_radius)  # in Newtons
    return total_force


# Part 2: Contact Pressure Functions

def check_existing_contact_pressure_data(speed_folder_path):
    """
    Check if contact pressure data already exists in output/piston folder
    Returns (max_pressure, mean_pressure) if found, (None, None) if not found
    """
    try:
        # Check for existing contact pressure results in output/piston folder
        output_piston_folder = os.path.join(speed_folder_path, 'output', 'piston')

        # Look for files that might contain pre-calculated contact pressure data
        potential_files = [
            'contact_pressure_results.txt',
            'contact_pressure_summary.csv',
            'cumulative_pressure_results.txt',
            'pressure_analysis.txt'
        ]

        for filename in potential_files:
            filepath = os.path.join(output_piston_folder, filename)
            if os.path.exists(filepath):
                print(f"    Found existing contact pressure data: {filename}")
                try:
                    df = read_data_file(filepath)
                    if df is not None:
                        # Look for cumulative pressure columns
                        pressure_cols = []
                        for col in df.columns:
                            if any(keyword in col.lower() for keyword in ['cumulative', 'contact', 'pressure']):
                                pressure_cols.append(col)

                        if pressure_cols:
                            best_col = pressure_cols[0]  # Use first found column
                            max_pressure = float(df[best_col].max())
                            mean_pressure = float(df[best_col].mean())
                            print(
                                f"    Loaded existing data - Max: {max_pressure:.2e} Pa, Mean: {mean_pressure:.2e} Pa")
                            return max_pressure, mean_pressure
                except Exception as e:
                    print(f"    Error reading existing file {filename}: {e}")
                    continue

        return None, None

    except Exception as e:
        print(f"    Error checking existing contact pressure data: {e}")
        return None, None


def read_contact_pressure_data_t1(speed_folder_path, speed_val):
    """
    Read contact pressure data using T1 functions for cumulative pressure analysis
    First checks for existing data, then runs T1 analysis if needed
    Returns max and mean cumulative contact pressure if available
    """
    try:
        print(f"    Checking for existing contact pressure data for speed {speed_val}...")

        # First, check if data already exists
        existing_max, existing_mean = check_existing_contact_pressure_data(speed_folder_path)
        if existing_max is not None and existing_mean is not None:
            print(f"    Using existing contact pressure data - skipping T1 analysis")
            return existing_max, existing_mean

        print(f"    No existing data found, attempting T1 contact pressure analysis...")

        # Check if required files exist for T1 analysis
        geom_file = os.path.join(speed_folder_path, 'input', 'geometry.txt')
        op_file = os.path.join(speed_folder_path, 'input', 'operatingconditions.txt')
        matlab_file = os.path.join(speed_folder_path, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt')

        if not all(os.path.exists(f) for f in [geom_file, op_file, matlab_file]):
            print(f"    Missing required files for T1 analysis")
            print(f"      Geometry file: {os.path.exists(geom_file)}")
            print(f"      Operating conditions file: {os.path.exists(op_file)}")
            print(f"      Matlab contact pressure file: {os.path.exists(matlab_file)}")
            return None, None

        # Parse geometry and operating conditions
        lF_val = parse_geometry(geom_file)
        speed_val_parsed, hp_val = parse_operating_conditions(op_file)

        if lF_val is None or speed_val_parsed is None or hp_val is None:
            print(f"    Could not parse geometry or operating conditions")
            return None, None

        print(f"    T1 Parameters: lF={lF_val:.1f}mm, speed={speed_val_parsed:.0f}rpm, ΔP={hp_val:.1f}")

        # Run T1 contact pressure analysis
        result = create_cumulative_pressure_map(
            filepath=speed_folder_path,
            m=50,  # Standard value from T1
            lF=lF_val,
            n=speed_val_parsed,
            deltap=hp_val,
            plots=360,  # Standard value from T1
            degstep=1,  # Standard value from T1
            ignore=0,  # Standard value from T1
            offset=0,  # Standard value from T1
            minplot=0,  # Auto-calculate
            maxplot=0  # Auto-calculate
        )

        if result is not None and 'cumulative' in result:
            mean_cumulative_pressure = np.mean(result['cumulative'])
            max_cumulative_pressure = np.max(result['cumulative'])
            print(f"    T1 Success - Mean: {mean_cumulative_pressure:.2e} Pa, Max: {max_cumulative_pressure:.2e} Pa")

            # Save the results for future use
            try:
                output_file = os.path.join(speed_folder_path, 'output', 'piston', 'contact_pressure_results.txt')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write("max_cumulative_pressure\tmean_cumulative_pressure\n")
                    f.write(f"{max_cumulative_pressure}\t{mean_cumulative_pressure}\n")
                print(f"    Saved contact pressure results to: {output_file}")
            except Exception as e:
                print(f"    Warning: Could not save contact pressure results: {e}")

            return max_cumulative_pressure, mean_cumulative_pressure
        else:
            print(f"    T1 analysis returned None or missing 'cumulative' key")
            return None, None

    except Exception as e:
        print(f"    T1 Error: {str(e)[:100]}...")
        return None, None


def read_contact_pressure_data_fallback(speed_folder_path):
    """
    Fallback method to extract contact pressure from piston.txt if T1 analysis fails
    """
    try:
        print(f"    Attempting fallback contact pressure analysis...")
        piston_file = os.path.join(speed_folder_path, 'output', 'piston', 'piston.txt')

        if not os.path.exists(piston_file):
            print(f"    Piston file not found: {piston_file}")
            return None, None

        df = read_data_file(piston_file)
        if df is None:
            print(f"    Could not read piston file")
            return None, None

        # Look for contact pressure related columns
        potential_cp_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['contact', 'pressure', 'force']):
                if not any(exclude in col_lower for exclude in ['loss', 'power', 'volume']):
                    potential_cp_cols.append(col)

        if potential_cp_cols:
            # Use revolution data (similar to power loss analysis)
            df_filtered = df[df['revolution'] <= 6.0] if 'revolution' in df.columns else df

            best_col = None
            for col in potential_cp_cols:
                if 'contact' in col.lower():
                    best_col = col
                    break
            if not best_col:
                best_col = potential_cp_cols[0]

            if df_filtered[best_col].notna().any():
                max_pressure = float(df_filtered[best_col].max())
                mean_pressure = float(df_filtered[best_col].mean())
                print(f"    Fallback Success ({best_col}) - Mean: {mean_pressure:.2e} Pa, Max: {max_pressure:.2e} Pa")
                return max_pressure, mean_pressure

        print(f"    No contact pressure columns found in fallback")
        return None, None

    except Exception as e:
        print(f"    Fallback Error: {str(e)[:100]}...")
        return None, None


# Part 3: Data Processing Function

def process_multiple_folder_structure_with_contact_pressure(folder_paths):
    """
    Enhanced version that collects both power loss and contact pressure data
    """
    all_data = {}

    for folder_name, folder_path in folder_paths:
        print(f"\n=== Processing {folder_name}: {folder_path} ===")

        if not os.path.exists(folder_path):
            print(f"Error: Path does not exist: {folder_path}")
            continue

        folder_data = {}

        # List all items in folder
        folder_items = os.listdir(folder_path)
        print(f"Found items in {folder_name}: {folder_items}")

        # Look for speed folders
        speed_folders = []
        for item in folder_items:
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                speed_folders.append(int(item))

        speed_folders.sort()
        print(f"Found speed folders in {folder_name}: {speed_folders}")

        # Process each speed folder
        for speed_val in speed_folders:
            speed_folder_path = os.path.join(folder_path, str(speed_val))
            print(f"\n  Processing {folder_name} - Speed: {speed_val}")

            # Read power loss data
            piston_file = os.path.join(speed_folder_path, 'output', 'piston', 'piston.txt')

            if os.path.exists(piston_file):
                print(f"    Found piston file for power loss analysis")
                mech_loss, vol_loss = read_piston_data(piston_file)

                if mech_loss is not None and vol_loss is not None:
                    # Calculate total force
                    total_force = calculate_total_force(mech_loss, vol_loss, speed_val)

                    # Read contact pressure data using T1 method first
                    max_contact_pressure, mean_contact_pressure = read_contact_pressure_data_t1(speed_folder_path,
                                                                                                speed_val)

                    # If T1 fails, try fallback method
                    if max_contact_pressure is None:
                        max_contact_pressure, mean_contact_pressure = read_contact_pressure_data_fallback(
                            speed_folder_path)

                    # Store all data
                    folder_data[speed_val] = {
                        'mechanical_loss': mech_loss,
                        'volumetric_loss': vol_loss,
                        'total_loss': mech_loss + vol_loss,
                        'total_force': total_force,
                        'max_contact_pressure': max_contact_pressure,
                        'mean_contact_pressure': mean_contact_pressure,
                        'speed': speed_val
                    }

                    print(
                        f"    Power Loss - Mech: {mech_loss:.2f}W, Vol: {vol_loss:.2f}W, Total: {mech_loss + vol_loss:.2f}W")
                    print(f"    Total Force: {total_force:.2f}N")
                    if max_contact_pressure is not None:
                        print(
                            f"    Contact Pressure - Max: {max_contact_pressure:.2e}Pa, Mean: {mean_contact_pressure:.2e}Pa")
                    else:
                        print(f"    Contact Pressure: Not available")
                else:
                    print(f"    Failed to read power loss data")
            else:
                print(f"    Piston file not found: {piston_file}")

        all_data[folder_name] = folder_data

    return all_data


# Part 4: Plotting Functions

def create_comparative_power_loss_plot(all_data, relative=True):
    """
    Create a comparative plot showing power losses from multiple folders (supports 4 folders)
    Shows difference from Standard-Design baseline above each bar
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.figsize': (10, 8),
        'font.family': 'serif'
    })

    # Extract all speeds across all folders
    all_speeds = set()
    for folder_data in all_data.values():
        all_speeds.update(folder_data.keys())

    speeds = sorted(list(all_speeds))
    folder_names = list(all_data.keys())

    print(f"Speeds: {speeds}")
    print(f"Folders: {folder_names}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors for different folders - expanded for 4 folders
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#42B883']  # Blue, Purple, Orange, Green
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}

    # Prepare data
    x = np.arange(len(speeds))
    width = 0.8 / len(folder_names)  # Adjust width based on number of folders

    # Get data for all folders
    all_folder_data = {}
    for folder_name in folder_names:
        folder_data = []
        for speed in speeds:
            if speed in all_data.get(folder_name, {}):
                total_loss = all_data[folder_name][speed]['total_loss']
                folder_data.append(total_loss)
            else:
                folder_data.append(0)
        all_folder_data[folder_name] = folder_data

    # Find the Standard-Design baseline data for difference calculations
    standard_design_key = None
    for folder_name in folder_names:
        if 'Standard-Design' in folder_name or 'standard' in folder_name.lower():
            standard_design_key = folder_name
            break

    # If no standard design found, use the last folder as baseline
    if standard_design_key is None:
        standard_design_key = folder_names[-1]  # Use last folder as baseline
        print(f"Warning: No 'Standard-Design' folder found. Using '{standard_design_key}' as baseline.")

    standard_design_data = all_folder_data[standard_design_key]

    # Convert to relative percentages if requested
    if relative:
        total_sum = sum(sum(data) for data in all_folder_data.values())
        if total_sum > 0:
            for folder_name in folder_names:
                all_folder_data[folder_name] = [(val / total_sum) * 100 for val in all_folder_data[folder_name]]
            # Recalculate standard design data after percentage conversion
            standard_design_data = all_folder_data[standard_design_key]

    # Create bars for each folder
    for i, folder_name in enumerate(folder_names):
        folder_data = all_folder_data[folder_name]

        # Position for this folder's bars
        pos = x + (i - len(folder_names) / 2 + 0.5) * width

        bars = ax.bar(pos, folder_data, width,
                      label=folder_name, color=folder_colors[folder_name], alpha=0.8)

        # Add difference labels on bars (difference from standard design)
        for j, (bar, val) in enumerate(zip(bars, folder_data)):
            if val > 0:
                # Calculate difference from standard design for this speed
                baseline_val = standard_design_data[j] if j < len(standard_design_data) else 0

                if folder_name == standard_design_key:
                    # For standard design itself, don't show any label
                    continue
                else:
                    # For other methods, show difference from standard design
                    difference = val - baseline_val

                    if relative:
                        # For relative plots, show percentage point difference
                        if difference > 0:
                            label_text = f'+{difference:.1f}%'  # pp = percentage points
                        else:
                            label_text = f'{difference:.1f}%'
                    else:
                        # For absolute plots, show difference in watts
                        if difference > 0:
                            label_text = f'+{difference:.0f}W'
                        else:
                            label_text = f'{difference:.0f}W'

                    # Position label above bar with reduced gap
                    if relative:
                        y_offset = 0.1  # Reduced from 0.5
                    else:
                        max_val = max(max(data) for data in all_folder_data.values())
                        y_offset = max_val * 0.002  # Reduced from 0.01

                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                            label_text, ha='center', va='bottom',
                            fontweight='bold', fontsize=14, rotation=0)

    # Formatting
    ax.set_xlabel('Speed (RPM)', fontweight='bold')
    if relative:
        ax.set_ylabel('Power Loss (%)', fontweight='bold')
    else:
        ax.set_ylabel('Power Loss (W)', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([str(speed) for speed in speeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Title
    title_prefix = "Relative " if relative else ""
    ax.set_title(f'{title_prefix}Power Loss Comparison: Optimization Methods\n(Pressure:320bar, Displacement:100%)',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()

    # Save the figure
    fig.savefig('comparative_power_loss_analysis_with_differences.png', dpi=300, bbox_inches='tight')

    return fig, ax


def create_contact_pressure_vs_speed_bar_plot(all_data, show_differences=True):
    """
    Create a comparative bar plot showing Max Contact Pressure vs Speed from multiple folders
    Similar to the power loss vs speed bar plots but for contact pressure (LINEAR SCALE ONLY)
    Shows difference from Standard-Design baseline above each bar
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.figsize': (10, 8),
        'font.family': 'serif'
    })

    # Extract all speeds across all folders
    all_speeds = set()
    for folder_data in all_data.values():
        all_speeds.update(folder_data.keys())

    speeds = sorted(list(all_speeds))
    folder_names = list(all_data.keys())

    print(f"Speeds: {speeds}")
    print(f"Folders: {folder_names}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors for different folders - expanded for 4 folders
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#42B883']  # Blue, Purple, Orange, Green
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}

    # Prepare data
    x = np.arange(len(speeds))
    width = 0.8 / len(folder_names)  # Adjust width based on number of folders

    # Get data for all folders
    all_folder_data = {}
    for folder_name in folder_names:
        folder_data = []
        for speed in speeds:
            if speed in all_data.get(folder_name, {}):
                max_contact_pressure = all_data[folder_name][speed]['max_contact_pressure']
                if max_contact_pressure is not None:
                    folder_data.append(max_contact_pressure)
                else:
                    folder_data.append(0)
            else:
                folder_data.append(0)
        all_folder_data[folder_name] = folder_data

    # Find the Standard-Design baseline data for difference calculations
    standard_design_key = None
    for folder_name in folder_names:
        if 'Standard-Design' in folder_name or 'standard' in folder_name.lower():
            standard_design_key = folder_name
            break

    # If no standard design found, use the last folder as baseline
    if standard_design_key is None:
        standard_design_key = folder_names[-1]  # Use last folder as baseline
        print(f"Warning: No 'Standard-Design' folder found. Using '{standard_design_key}' as baseline.")

    standard_design_data = all_folder_data[standard_design_key]

    # Check if we have any valid data
    has_data = any(any(val > 0 for val in data) for data in all_folder_data.values())

    if not has_data:
        # Create placeholder plot with message
        ax.text(0.5, 0.5, 'No Contact Pressure Data Available\nfor Any Speed',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=16, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Speed (RPM)', fontweight='bold')
        ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
        ax.set_title('Max Cumulative Contact Pressure vs Speed\n(Pressure: 320bar, Displacement: 100%)',
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        return fig, ax, {}

    # Create bars for each folder
    for i, folder_name in enumerate(folder_names):
        folder_data = all_folder_data[folder_name]

        # Position for this folder's bars
        pos = x + (i - len(folder_names) / 2 + 0.5) * width

        bars = ax.bar(pos, folder_data, width,
                      label=folder_name, color=folder_colors[folder_name], alpha=0.8)

        # Add difference labels on bars (difference from standard design)
        if show_differences:
            for j, (bar, val) in enumerate(zip(bars, folder_data)):
                if val > 0:
                    # Calculate difference from standard design for this speed
                    baseline_val = standard_design_data[j] if j < len(standard_design_data) else 0

                    if folder_name == standard_design_key:
                        # For standard design itself, show the actual value
                        label_text = f'{val:.1e}'
                    else:
                        # For other methods, show difference from standard design
                        if baseline_val > 0:
                            difference = val - baseline_val
                            percentage_diff = (difference / baseline_val) * 100

                            # Show percentage difference for better readability
                            if percentage_diff > 0:
                                label_text = f'+{percentage_diff:.1f}%'
                            else:
                                label_text = f'{percentage_diff:.1f}%'
                        else:
                            # If baseline is 0, just show the value
                            label_text = f'{val:.1e}'

                    # Position label above bar with reduced gap
                    max_val = max(max(data) for data in all_folder_data.values() if max(data) > 0)
                    y_offset = max_val * 0.02

                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                            label_text, ha='center', va='bottom',
                            fontweight='bold', fontsize=10, rotation=0)

    # Formatting
    ax.set_xlabel('Speed (RPM)', fontweight='bold')
    ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([str(speed) for speed in speeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Title
    ax.set_title('Max Cumulative Contact Pressure vs Speed\n(Pressure: 320bar, Displacement: 100%)',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()

    # Save the figure
    filename = 'contact_pressure_vs_speed_bar_plot_linear.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Contact pressure vs speed bar plot saved as: {filename}")

    return fig, ax, all_folder_data


def create_contact_pressure_vs_volumetric_loss_plot(all_data, target_speed=4500):
    """
    Create a plot showing Max Cumulative Contact Pressure vs Volumetric Loss
    for the specified speed across all optimization methods (LINEAR SCALE)
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.figsize': (10, 8),
        'font.family': 'serif'
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors and markers for different folders
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#42B883']  # Blue, Purple, Orange, Green
    markers = ['o', '^', 's', 'D']  # Circle, Triangle, Square, Diamond

    folder_names = list(all_data.keys())
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}
    folder_markers = {folder: markers[i % len(markers)] for i, folder in enumerate(folder_names)}

    plot_data = []
    valid_data_found = False

    for folder_name, folder_data in all_data.items():
        if target_speed in folder_data:
            speed_data = folder_data[target_speed]

            volumetric_loss = speed_data['volumetric_loss']
            max_contact_pressure = speed_data['max_contact_pressure']

            if max_contact_pressure is not None and volumetric_loss is not None:
                plot_data.append({
                    'folder_name': folder_name,
                    'volumetric_loss': volumetric_loss,
                    'max_contact_pressure': max_contact_pressure
                })

                # Plot point
                ax.scatter(volumetric_loss, max_contact_pressure,
                           color=folder_colors[folder_name],
                           marker=folder_markers[folder_name],
                           s=150, alpha=0.8,
                           label=folder_name,
                           edgecolor='black', linewidth=1)

                # Add text label
                ax.annotate(folder_name,
                            (volumetric_loss, max_contact_pressure),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=12, fontweight='bold')

                valid_data_found = True
            else:
                print(f"No complete data available for {folder_name} at {target_speed} RPM")

    if not valid_data_found:
        # Create placeholder plot with message
        ax.text(0.5, 0.5, 'No Contact Pressure or Volumetric Loss Data Available\nfor the Specified Speed',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=16, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Volumetric Loss [W]', fontweight='bold')
        ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
    else:
        # Formatting - CHANGED TO LINEAR SCALE
        ax.set_xlabel('Volumetric Loss [W]', fontweight='bold')
        ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
        # ax.set_yscale('log')  # REMOVED - Now using linear scale
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax.set_title(f'Max Cumulative Contact Pressure vs Volumetric Loss\n(Speed: {target_speed} RPM)',
                 fontweight='bold', fontsize=16)

    plt.tight_layout()
    fig.savefig(f'contact_pressure_vs_volumetric_loss_{target_speed}rpm_linear.png', dpi=300,
                bbox_inches='tight')

    return fig, ax, plot_data


def create_contact_pressure_vs_total_loss_plot(all_data, target_speed=4500):
    """
    Create a plot showing Max Cumulative Contact Pressure vs Total Loss
    for the specified speed across all optimization methods (LINEAR SCALE)
    """
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 16,
        'legend.fontsize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'figure.figsize': (10, 8),
        'font.family': 'serif'
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors and markers for different folders
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', '^', 's']

    folder_names = list(all_data.keys())
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}
    folder_markers = {folder: markers[i % len(markers)] for i, folder in enumerate(folder_names)}

    plot_data = []
    valid_data_found = False

    for folder_name, folder_data in all_data.items():
        if target_speed in folder_data:
            speed_data = folder_data[target_speed]

            total_loss = speed_data['total_loss']
            max_contact_pressure = speed_data['max_contact_pressure']

            if max_contact_pressure is not None:
                plot_data.append({
                    'folder_name': folder_name,
                    'total_loss': total_loss,
                    'max_contact_pressure': max_contact_pressure
                })

                # Plot point
                ax.scatter(total_loss, max_contact_pressure,
                           color=folder_colors[folder_name],
                           marker=folder_markers[folder_name],
                           s=150, alpha=0.8,
                           label=folder_name,
                           edgecolor='black', linewidth=1)

                # Add text label
                ax.annotate(folder_name,
                            (total_loss, max_contact_pressure),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=16, fontweight='bold')

                valid_data_found = True
            else:
                print(f"No contact pressure data available for {folder_name} at {target_speed} RPM")

    if not valid_data_found:
        # Create placeholder plot with message
        ax.text(0.5, 0.5, 'No Contact Pressure Data Available\nfor the Specified Speed',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=16, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Total Loss [W]', fontweight='bold')
        ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
    else:
        # Formatting - CHANGED TO LINEAR SCALE
        ax.set_xlabel('Total Loss [W]', fontweight='bold')
        ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
        # ax.set_yscale('log')  # REMOVED - Now using linear scale
        # ax.set_xscale('log')  # REMOVED - Now using linear scale
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax.set_title(f'Max Cumulative Contact Pressure vs Total Loss\n(Speed: {target_speed} RPM)',
                 fontweight='bold', fontsize=16)

    plt.tight_layout()
    fig.savefig(f'contact_pressure_vs_total_loss_{target_speed}rpm_linear.png', dpi=300,
                bbox_inches='tight')

    return fig, ax, plot_data


def create_contact_pressure_vs_hydromechanical_loss_plot(all_data, target_speed=4500):
    """
    Create a plot showing Max Cumulative Contact Pressure vs Hydromechanical Loss
    for the specified speed across all optimization methods (LINEAR SCALE).
    Hydromechanical Loss = Mechanical Loss + Volumetric Loss (Total Loss)
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.figsize': (10, 8),
        'font.family': 'serif'
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors and markers for different folders
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#42B883']  # Blue, Purple, Orange, Green
    markers = ['o', '^', 's', 'D']  # Circle, Triangle, Square, Diamond

    folder_names = list(all_data.keys())
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}
    folder_markers = {folder: markers[i % len(markers)] for i, folder in enumerate(folder_names)}

    plot_data = []
    valid_data_found = False

    for folder_name, folder_data in all_data.items():
        if target_speed in folder_data:
            speed_data = folder_data[target_speed]

            # Hydromechanical loss is the mechanical loss only
            hydromechanical_loss = speed_data['mechanical_loss']
            max_contact_pressure = speed_data['max_contact_pressure']

            if max_contact_pressure is not None and hydromechanical_loss is not None:
                plot_data.append({
                    'folder_name': folder_name,
                    'hydromechanical_loss': hydromechanical_loss,
                    'max_contact_pressure': max_contact_pressure,
                    'mechanical_loss': speed_data['mechanical_loss'],
                    'volumetric_loss': speed_data['volumetric_loss']
                })

                # Plot point
                ax.scatter(hydromechanical_loss, max_contact_pressure,
                           color=folder_colors[folder_name],
                           marker=folder_markers[folder_name],
                           s=150, alpha=0.8,
                           label=folder_name,
                           edgecolor='black', linewidth=1)

                # Add text label
                ax.annotate(folder_name,
                            (hydromechanical_loss, max_contact_pressure),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=12, fontweight='bold')

                valid_data_found = True
            else:
                print(f"No complete data available for {folder_name} at {target_speed} RPM")

    if not valid_data_found:
        # Create placeholder plot with message
        ax.text(0.5, 0.5, 'No Contact Pressure or Hydromechanical Loss Data Available\nfor the Specified Speed',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=16, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Hydromechanical Loss [W]', fontweight='bold')
        ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
    else:
        # Formatting - CHANGED TO LINEAR SCALE
        ax.set_xlabel('Mechanical Loss [W]', fontweight='bold')
        ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
        # ax.set_yscale('log')  # REMOVED - Now using linear scale
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax.set_title(
        f'Max Cumulative Contact Pressure vs Mechanical Loss\n(Speed: {target_speed} RPM, Pressure:320bar, Displacement: 100%)',
        fontweight='bold', fontsize=16)

    plt.tight_layout()
    fig.savefig(f'contact_pressure_vs_hydromechanical_loss_{target_speed}rpm_linear.png', dpi=300,
                bbox_inches='tight')

    return fig, ax, plot_data


def create_dual_column_plot_volumetric(all_data, target_speed=4500):
    """
    Create a dual-column plot for 3 folders:
    Left: Total Loss Bar Chart for target speed
    Right: Contact Pressure vs Volumetric Loss Scatter Plot for target speed (LINEAR SCALE)
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'figure.figsize': (16, 8),
        'font.family': 'serif'
    })

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Colors for different folders
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    folder_names = list(all_data.keys())
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}

    # Prepare data for target speed
    total_losses = []
    plot_data = []
    volumetric_losses = []
    contact_pressures = []
    folder_labels = []
    bar_colors = []

    for folder_name, folder_data in all_data.items():
        if target_speed in folder_data:
            speed_data = folder_data[target_speed]
            volumetric_loss = speed_data['volumetric_loss']
            total_loss = speed_data['total_loss']
            max_contact_pressure = speed_data['max_contact_pressure']

            if max_contact_pressure is not None and volumetric_loss is not None:
                plot_data.append({
                    'folder_name': folder_name,
                    'volumetric_loss': volumetric_loss,
                    'max_contact_pressure': max_contact_pressure,
                    'total_loss': total_loss
                })

                total_losses.append(total_loss)
                volumetric_losses.append(volumetric_loss)
                contact_pressures.append(max_contact_pressure)
                folder_labels.append(folder_name)
                bar_colors.append(folder_colors[folder_name])

    if not plot_data:
        # Handle case where no data is available
        ax1.text(0.5, 0.5, 'No Data Available\nfor Target Speed',
                 transform=ax1.transAxes, ha='center', va='center',
                 fontsize=14, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        ax2.text(0.5, 0.5, 'No Contact Pressure Data Available\nfor Target Speed',
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=14, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

        ax1.set_title(f'Total Loss Comparison\n(Speed: {target_speed} RPM, Pressure: 320bar, Displacement: 100%)',
                      fontweight='bold', fontsize=14)
        ax2.set_title(
            f'Max Contact Pressure vs Volumetric Loss\n(Speed: {target_speed} RPM, Pressure: 320bar, Displacement: 100%)',
            fontweight='bold',
            fontsize=14)

        plt.tight_layout()
        return fig, (ax1, ax2), []

    # LEFT SUBPLOT: Bar chart of total losses
    bars = ax1.bar(folder_labels, total_losses, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels on bars
    max_loss = max(total_losses)
    for bar, loss in zip(bars, total_losses):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_loss * 0.02,
                 f'{loss:.0f}W', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax1.set_xlabel('Optimization Method', fontweight='bold')
    ax1.set_ylabel('Total Loss [W]', fontweight='bold')
    ax1.set_title(f'Total Loss Comparison\n(Speed: {target_speed} RPM, Pressure: 320bar, Displacement: 100%)',
                  fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max_loss * 1.15)

    # Rotate x-axis labels if they're long
    if any(len(label) > 10 for label in folder_labels):
        ax1.tick_params(axis='x', rotation=45)

    # RIGHT SUBPLOT: Scatter plot of contact pressure vs volumetric loss (LINEAR SCALE)
    markers = ['o', '^', 's']  # Circle, Triangle, Square
    folder_markers = {folder: markers[i % len(markers)] for i, folder in enumerate(folder_names)}

    for i, data in enumerate(plot_data):
        folder_name = data['folder_name']
        volumetric_loss = data['volumetric_loss']
        max_contact_pressure = data['max_contact_pressure']

        ax2.scatter(volumetric_loss, max_contact_pressure,
                    color=folder_colors[folder_name],
                    marker=folder_markers[folder_name],
                    s=200, alpha=0.8,
                    label=folder_name,
                    edgecolor='black', linewidth=1)

    ax2.set_xlabel('Volumetric Loss [W]', fontweight='bold')
    ax2.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
    # ax2.set_yscale('log')  # REMOVED - Now using linear scale
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title(
        f'Max Contact Pressure vs Volumetric Loss\n(Speed: {target_speed} RPM, Pressure: 320bar, Displacement: 100%)',
        fontweight='bold', fontsize=14)

    plt.tight_layout()

    # Save the plot
    filename = f'dual_column_volumetric_analysis_{target_speed}rpm_linear.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Dual-column volumetric plot (linear scale) saved as: {filename}")

    return fig, (ax1, ax2), plot_data


def create_detailed_breakdown_plot(all_data):
    """
    Create a detailed breakdown plot showing mechanical vs volumetric losses
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (8, 8),
        'font.family': 'serif'
    })

    # Extract all speeds across all folders
    all_speeds = set()
    for folder_data in all_data.values():
        all_speeds.update(folder_data.keys())

    speeds = sorted(list(all_speeds))
    folder_names = list(all_data.keys())

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Colors for different folders
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}

    # Prepare data
    x = np.arange(len(speeds))
    width = 0.8 / (len(folder_names) * 2)

    # Get mechanical and volumetric data for all folders
    all_folder_mech_data = {}
    all_folder_vol_data = {}

    for folder_name in folder_names:
        mech_data = []
        vol_data = []
        for speed in speeds:
            if speed in all_data.get(folder_name, {}):
                mech_data.append(all_data[folder_name][speed]['mechanical_loss'])
                vol_data.append(all_data[folder_name][speed]['volumetric_loss'])
            else:
                mech_data.append(0)
                vol_data.append(0)
        all_folder_mech_data[folder_name] = mech_data
        all_folder_vol_data[folder_name] = vol_data

    # Create grouped bars for each folder
    current_pos = 0

    for i, folder_name in enumerate(folder_names):
        mech_data = all_folder_mech_data[folder_name]
        vol_data = all_folder_vol_data[folder_name]

        # Positions
        mech_pos = x + current_pos * width
        vol_pos = x + (current_pos + 1) * width

        # Create bars
        bars_mech = ax.bar(mech_pos, mech_data, width,
                           label=f'{folder_name} - Mechanical',
                           color=folder_colors[folder_name], alpha=0.8)

        bars_vol = ax.bar(vol_pos, vol_data, width,
                          label=f'{folder_name} - Volumetric',
                          color=folder_colors[folder_name], alpha=0.5, hatch='///')

        # Add value labels on bars
        for bar, val in zip(bars_mech, mech_data):
            if val > 0:
                max_val = max(max(all_folder_mech_data[fn]) + max(all_folder_vol_data[fn]) for fn in folder_names)
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.01,
                        f'{val:.0f}W', ha='center', va='bottom',
                        fontweight='bold', fontsize=9, rotation=0)

        for bar, val in zip(bars_vol, vol_data):
            if val > 0:
                max_val = max(max(all_folder_mech_data[fn]) + max(all_folder_vol_data[fn]) for fn in folder_names)
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.01,
                        f'{val:.0f}W', ha='center', va='bottom',
                        fontweight='bold', fontsize=9, rotation=0)

        current_pos += 2.5

    # Formatting
    ax.set_xlabel('Speed (RPM)', fontweight='bold')
    ax.set_ylabel('Power Loss (W)', fontweight='bold')
    ax.set_title('Mechanical vs Volumetric Power Loss Comparison', fontweight='bold', fontsize=14)

    tick_offset = (len(folder_names) * 2.5 - 0.5) * width / 2
    ax.set_xticks(x + tick_offset)
    ax.set_xticklabels([str(speed) for speed in speeds])

    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig('detailed_breakdown_analysis_with_contact_pressure.png', dpi=300, bbox_inches='tight')

    return fig, ax


def create_dual_column_plot_volumetric(all_data, target_speed=4500):
    """
    Create a dual-column plot for 3 folders:
    Left: Total Loss Bar Chart for target speed
    Right: Contact Pressure vs Volumetric Loss Scatter Plot for target speed
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'figure.figsize': (16, 8),
        'font.family': 'serif'
    })

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Colors for different folders
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    folder_names = list(all_data.keys())
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}

    # Prepare data for target speed
    total_losses = []
    plot_data = []
    volumetric_losses = []
    contact_pressures = []
    folder_labels = []
    bar_colors = []

    for folder_name, folder_data in all_data.items():
        if target_speed in folder_data:
            speed_data = folder_data[target_speed]
            volumetric_loss = speed_data['volumetric_loss']
            total_loss = speed_data['total_loss']
            max_contact_pressure = speed_data['max_contact_pressure']

            if max_contact_pressure is not None and volumetric_loss is not None:
                plot_data.append({
                    'folder_name': folder_name,
                    'volumetric_loss': volumetric_loss,
                    'max_contact_pressure': max_contact_pressure,
                    'total_loss': total_loss
                })

                total_losses.append(total_loss)
                volumetric_losses.append(volumetric_loss)
                contact_pressures.append(max_contact_pressure)
                folder_labels.append(folder_name)
                bar_colors.append(folder_colors[folder_name])

    if not plot_data:
        # Handle case where no data is available
        ax1.text(0.5, 0.5, 'No Data Available\nfor Target Speed',
                 transform=ax1.transAxes, ha='center', va='center',
                 fontsize=14, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        ax2.text(0.5, 0.5, 'No Contact Pressure Data Available\nfor Target Speed',
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=14, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

        ax1.set_title(f'Total Loss Comparison\n(Speed: {target_speed} RPM, Pressure: 320bar, Displacement: 100%)',
                      fontweight='bold', fontsize=14)
        ax2.set_title(
            f'Max Contact Pressure vs Volumetric Loss\n(Speed: {target_speed} RPM, Pressure: 320bar, Displacement: 100%)',
            fontweight='bold',
            fontsize=14)

        plt.tight_layout()
        return fig, (ax1, ax2), []

    # LEFT SUBPLOT: Bar chart of total losses
    bars = ax1.bar(folder_labels, total_losses, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels on bars
    max_loss = max(total_losses)
    for bar, loss in zip(bars, total_losses):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_loss * 0.02,
                 f'{loss:.0f}W', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax1.set_xlabel('Optimization Method', fontweight='bold')
    ax1.set_ylabel('Total Loss [W]', fontweight='bold')
    ax1.set_title(f'Total Loss Comparison\n(Speed: {target_speed} RPM, Pressure: 320bar, Displacement: 100%)',
                  fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max_loss * 1.15)

    # Rotate x-axis labels if they're long
    if any(len(label) > 10 for label in folder_labels):
        ax1.tick_params(axis='x', rotation=45)

    # RIGHT SUBPLOT: Scatter plot of contact pressure vs volumetric loss
    markers = ['o', '^', 's']  # Circle, Triangle, Square
    folder_markers = {folder: markers[i % len(markers)] for i, folder in enumerate(folder_names)}

    for i, data in enumerate(plot_data):
        folder_name = data['folder_name']
        volumetric_loss = data['volumetric_loss']
        max_contact_pressure = data['max_contact_pressure']

        ax2.scatter(volumetric_loss, max_contact_pressure,
                    color=folder_colors[folder_name],
                    marker=folder_markers[folder_name],
                    s=200, alpha=0.8,
                    label=folder_name,
                    edgecolor='black', linewidth=1)

    ax2.set_xlabel('Volumetric Loss [W]', fontweight='bold')
    ax2.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title(
        f'Max Contact Pressure vs Volumetric Loss\n(Speed: {target_speed} RPM, Pressure: 320bar, Displacement: 100%)',
        fontweight='bold', fontsize=14)

    plt.tight_layout()

    # Save the plot
    filename = f'dual_column_volumetric_analysis_{target_speed}rpm.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Dual-column volumetric plot saved as: {filename}")

    return fig, (ax1, ax2), plot_data


def create_contact_pressure_vs_hydromechanical_loss_plot(all_data, target_speed=4500):
    """
    Create a plot showing Max Cumulative Contact Pressure vs Hydromechanical Loss
    for the specified speed across all optimization methods.
    Hydromechanical Loss = Mechanical Loss + Volumetric Loss (Total Loss)
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.figsize': (10, 8),
        'font.family': 'serif'
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors and markers for different folders
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#42B883']  # Blue, Purple, Orange, Green
    markers = ['o', '^', 's', 'D']  # Circle, Triangle, Square, Diamond

    folder_names = list(all_data.keys())
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}
    folder_markers = {folder: markers[i % len(markers)] for i, folder in enumerate(folder_names)}

    plot_data = []
    valid_data_found = False

    for folder_name, folder_data in all_data.items():
        if target_speed in folder_data:
            speed_data = folder_data[target_speed]

            # Hydromechanical loss is the mechanical loss only
            hydromechanical_loss = speed_data['mechanical_loss']
            max_contact_pressure = speed_data['max_contact_pressure']

            if max_contact_pressure is not None and hydromechanical_loss is not None:
                plot_data.append({
                    'folder_name': folder_name,
                    'hydromechanical_loss': hydromechanical_loss,
                    'max_contact_pressure': max_contact_pressure,
                    'mechanical_loss': speed_data['mechanical_loss'],
                    'volumetric_loss': speed_data['volumetric_loss']
                })

                # Plot point
                ax.scatter(hydromechanical_loss, max_contact_pressure,
                           color=folder_colors[folder_name],
                           marker=folder_markers[folder_name],
                           s=150, alpha=0.8,
                           label=folder_name,
                           edgecolor='black', linewidth=1)

                # Add text label
                ax.annotate(folder_name,
                            (hydromechanical_loss, max_contact_pressure),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=12, fontweight='bold')

                valid_data_found = True
            else:
                print(f"No complete data available for {folder_name} at {target_speed} RPM")

    if not valid_data_found:
        # Create placeholder plot with message
        ax.text(0.5, 0.5, 'No Contact Pressure or Hydromechanical Loss Data Available\nfor the Specified Speed',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=16, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Hydromechanical Loss [W]', fontweight='bold')
        ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
    else:
        # Formatting
        ax.set_xlabel('Mechanical Loss [W]', fontweight='bold')
        ax.set_ylabel('Max Cumulative Contact Pressure [Pa]', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax.set_title(
        f'Max Cumulative Contact Pressure vs Mechanical Loss\n(Speed: {target_speed} RPM, Pressure:320bar, Displacement: 100%)',
        fontweight='bold', fontsize=16)

    plt.tight_layout()
    fig.savefig(f'contact_pressure_vs_hydromechanical_loss_{target_speed}rpm.png', dpi=300,
                bbox_inches='tight')

    return fig, ax, plot_data


# Part 5: Summary and Main Functions

def create_summary_comparison_table(all_data):
    """
    Create a comprehensive summary table including contact pressure data
    """
    print("\n" + "=" * 140)
    print("COMPREHENSIVE SUMMARY COMPARISON TABLE (Power Loss + Contact Pressure)")
    print("=" * 140)

    # Get all speeds and folder names
    all_speeds = set()
    for folder_data in all_data.values():
        all_speeds.update(folder_data.keys())

    speeds = sorted(list(all_speeds))
    folder_names = list(all_data.keys())

    # Create header for power loss
    header = f"{'Speed (RPM)':<12}"
    for folder_name in folder_names:
        header += f" {folder_name + ' Loss(W)':<20}"
        header += f" {folder_name + ' Force(N)':<20}"
        header += f" {folder_name + ' MaxCP(Pa)':<20}"

    print(header)
    print("-" * len(header))

    for speed in speeds:
        row = f"{speed:<12}"

        for folder_name in folder_names:
            if speed in all_data.get(folder_name, {}):
                data = all_data[folder_name][speed]
                total_loss = data['total_loss']
                total_force = data['total_force']
                max_cp = data['max_contact_pressure']

                row += f" {total_loss:<20.2f}"
                row += f" {total_force:<20.2f}"
                if max_cp is not None:
                    row += f" {max_cp:<20.2e}"
                else:
                    row += f" {'N/A':<20}"
            else:
                row += f" {0:<20.2f}"
                row += f" {0:<20.2f}"
                row += f" {'N/A':<20}"

        print(row)

    # Print detailed breakdown
    print("\n" + "=" * 140)
    print("DETAILED BREAKDOWN WITH CONTACT PRESSURE")
    print("=" * 140)

    for folder_name, folder_data in all_data.items():
        print(f"\n--- {folder_name} ---")
        print(
            f"{'Speed':<8} {'Mech(W)':<10} {'Vol(W)':<10} {'Total(W)':<10} {'Force(N)':<10} {'MaxCP(Pa)':<15} {'MeanCP(Pa)':<15}")
        print("-" * 80)

        for speed in sorted(folder_data.keys()):
            data = folder_data[speed]
            max_cp = data['max_contact_pressure']
            mean_cp = data['mean_contact_pressure']

            max_cp_str = f"{max_cp:.2e}" if max_cp is not None else "N/A"
            mean_cp_str = f"{mean_cp:.2e}" if mean_cp is not None else "N/A"

            print(f"{speed:<8} {data['mechanical_loss']:<10.2f} {data['volumetric_loss']:<10.2f} "
                  f"{data['total_loss']:<10.2f} {data['total_force']:<10.2f} "
                  f"{max_cp_str:<15} {mean_cp_str:<15}")


def main(folder_paths, plot_type='relative', target_speed=4500):
    """
    Main function to process multiple folders and create comparative visualization
    with integrated contact pressure analysis using T1 methods
    """
    print("Processing multiple-folder data structure with contact pressure analysis...")
    all_data = process_multiple_folder_structure_with_contact_pressure(folder_paths)

    if not all_data or not any(all_data.values()):
        print("No data found. Please check the folder structure and file paths.")
        print("Expected structure:")
        print("  folder_x/[speed_folders]/output/piston/piston.txt")
        print("  folder_x/[speed_folders]/input/geometry.txt")
        print("  folder_x/[speed_folders]/input/operatingconditions.txt")
        print("  folder_x/[speed_folders]/output/piston/matlab/Piston_Contact_Pressure.txt")
        return

    print(f"\nFound data for folders: {list(all_data.keys())}")

    # Print detailed summary with contact pressure
    create_summary_comparison_table(all_data)

    # Determine if we want relative (normalized) plots
    relative = (plot_type.lower() == 'relative')

    # PLOT 1: Create and show the comparative power loss plot
    print("\nCreating Comparative Power Loss Plot (Plot 1)...")
    try:
        fig1, ax1 = create_comparative_power_loss_plot(all_data, relative=relative)
        if fig1 is not None and ax1 is not None:
            plt.figure(fig1.number)
            plt.show()
            print("Comparative power loss plot displayed successfully")
        else:
            print("⚠️ Could not create comparative plot")
    except Exception as e:
        print(f"⚠️ Error creating comparative plot: {e}")
        import traceback
        traceback.print_exc()

    # PLOT 2: Create detailed breakdown plot
    print("\nCreating Detailed Breakdown Plot (Plot 2)...")
    try:
        fig2, ax2 = create_detailed_breakdown_plot(all_data)
        if fig2 is not None and ax2 is not None:
            plt.figure(fig2.number)
            plt.show()
            print("Detailed breakdown plot displayed successfully")
        else:
            print("⚠️ Could not create detailed breakdown plot")
    except Exception as e:
        print(f"⚠️ Error creating detailed breakdown plot: {e}")

    # PLOT 3: Create contact pressure vs total loss plot
    print(f"\nCreating Contact Pressure vs Total Loss Plot (Plot 3) for {target_speed} RPM...")
    try:
        fig3, ax3, contact_data = create_contact_pressure_vs_total_loss_plot(all_data, target_speed=target_speed)
        if fig3 is not None and ax3 is not None:
            plt.figure(fig3.number)
            plt.show()
            print("Contact Pressure vs Total Loss plot displayed successfully")

            # Print summary of contact pressure data
            if contact_data:
                print(f"\nContact Pressure Analysis Summary for {target_speed} RPM:")
                for data in contact_data:
                    print(f"  {data['folder_name']}: Total Loss={data['total_loss']:.2f}W, "
                          f"Max Contact Pressure={data['max_contact_pressure']:.2e}Pa")
            else:
                print(f"No contact pressure data available for {target_speed} RPM")
        else:
            print("⚠️ Could not create contact pressure vs total loss plot")
    except Exception as e:
        print(f"⚠️ Error creating contact pressure vs total loss plot: {e}")
        import traceback
        traceback.print_exc()

    # PLOT 4: Create dual-column plot with volumetric loss
    print(f"\nCreating Dual-Column Plot (Volumetric Focus) (Plot 4) for {target_speed} RPM...")
    try:
        fig4, axes4, dual_data = create_dual_column_plot_volumetric(all_data, target_speed=target_speed)
        if fig4 is not None and axes4 is not None:
            plt.figure(fig4.number)
            plt.show()
            print("Dual-column volumetric plot displayed successfully")

            # Print summary
            if dual_data:
                print(f"\nDual-Column Volumetric Analysis Summary for {target_speed} RPM:")
                for data in dual_data:
                    print(f"  {data['folder_name']}: Volumetric Loss={data['volumetric_loss']:.2f}W, "
                          f"Max Contact Pressure={data['max_contact_pressure']:.2e}Pa")
            else:
                print(f"No complete data available for dual-column volumetric plot at {target_speed} RPM")
        else:
            print("⚠️ Could not create dual-column volumetric plot")
    except Exception as e:
        print(f"⚠️ Error creating dual-column volumetric plot: {e}")
        import traceback
        traceback.print_exc()

    # PLOT 5: Create contact pressure vs volumetric loss plot
    print(f"\nCreating Contact Pressure vs Volumetric Loss Plot (Plot 5) for {target_speed} RPM...")
    try:
        fig5, ax5, volumetric_contact_data = create_contact_pressure_vs_volumetric_loss_plot(all_data,
                                                                                             target_speed=target_speed)
        if fig5 is not None and ax5 is not None:
            plt.figure(fig5.number)
            plt.show()
            print("Contact Pressure vs Volumetric Loss plot displayed successfully")

            # Print summary of volumetric loss vs contact pressure data
            if volumetric_contact_data:
                print(f"\nContact Pressure vs Volumetric Loss Analysis Summary for {target_speed} RPM:")
                for data in volumetric_contact_data:
                    print(f"  {data['folder_name']}: Volumetric Loss={data['volumetric_loss']:.2f}W, "
                          f"Max Contact Pressure={data['max_contact_pressure']:.2e}Pa")
            else:
                print(f"No complete volumetric loss and contact pressure data available for {target_speed} RPM")
        else:
            print("⚠️ Could not create contact pressure vs volumetric loss plot")
    except Exception as e:
        print(f"⚠️ Error creating contact pressure vs volumetric loss plot: {e}")
        import traceback
        traceback.print_exc()

    # PLOT 6: Create contact pressure vs hydromechanical loss plot
    print(f"\nCreating Contact Pressure vs Hydromechanical Loss Plot (Plot 6) for {target_speed} RPM...")
    try:
        fig6, ax6, hydromechanical_contact_data = create_contact_pressure_vs_hydromechanical_loss_plot(all_data,
                                                                                                       target_speed=target_speed)
        if fig6 is not None and ax6 is not None:
            plt.figure(fig6.number)
            plt.show()
            print("Contact Pressure vs Hydromechanical Loss plot displayed successfully")

            # Print summary of hydromechanical loss vs contact pressure data
            if hydromechanical_contact_data:
                print(f"\nContact Pressure vs Hydromechanical Loss Analysis Summary for {target_speed} RPM:")
                for data in hydromechanical_contact_data:
                    print(f"  {data['folder_name']}: Hydromechanical Loss={data['hydromechanical_loss']:.2f}W "
                          f"(Mech: {data['mechanical_loss']:.2f}W + Vol: {data['volumetric_loss']:.2f}W), "
                          f"Max Contact Pressure={data['max_contact_pressure']:.2e}Pa")
            else:
                print(f"No complete hydromechanical loss and contact pressure data available for {target_speed} RPM")
        else:
            print("⚠️ Could not create contact pressure vs hydromechanical loss plot")
    except Exception as e:
        print(f"⚠️ Error creating contact pressure vs hydromechanical loss plot: {e}")
        import traceback
        traceback.print_exc()

    # PLOT 7: Create contact pressure vs speed bar plot (LINEAR SCALE ONLY)
    print(f"\nCreating Contact Pressure vs Speed Bar Plot (Plot 7) - Linear Scale...")
    try:
        fig7, ax7, speed_contact_data = create_contact_pressure_vs_speed_bar_plot(all_data, show_differences=True)
        if fig7 is not None and ax7 is not None:
            plt.figure(fig7.number)
            plt.show()
            print("Contact Pressure vs Speed bar plot (linear scale) displayed successfully")

            # Print summary of speed vs contact pressure data
            if speed_contact_data:
                print(f"\nContact Pressure vs Speed Bar Analysis Summary:")
                for folder_name, data in speed_contact_data.items():
                    print(f"  {folder_name}: {[f'{val:.2e}' if val > 0 else 'N/A' for val in data]}")
            else:
                print(f"No complete speed vs contact pressure data available")
        else:
            print("⚠️ Could not create contact pressure vs speed bar plot")
    except Exception as e:
        print(f"⚠️ Error creating contact pressure vs speed bar plot: {e}")
        import traceback
        traceback.print_exc()


def main_two_folders(folder_x_path, folder_y_path, plot_type='relative', target_speed=4500):
    """Convenience function for two folders with proper labels"""
    folder_paths = [
        ("Optimal-Bayesian", folder_x_path),
        ("Optimal-NSGA-III", folder_y_path)
    ]
    main(folder_paths, plot_type, target_speed)


def main_three_folders(folder_x_path, folder_y_path, folder_z_path, plot_type='relative', target_speed=4500):
    """Convenience function for three folders with proper labels"""
    folder_paths = [
        ("Optimal-Bayesian", folder_x_path),
        ("Optimal-NSGA-III", folder_y_path),
        ("Standard-Design", folder_z_path)
    ]
    main(folder_paths, plot_type, target_speed)


# Part 6: Main Execution Block

if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION - Update these paths to your actual folders
    # ============================================================================

    # Folder paths - UPDATE THESE TO MATCH YOUR ACTUAL PATHS
    folder_x_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\standard_design\optimal_BO_3_para_dK_lKG_LF"
    folder_y_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\standard_design\optimal_nsga_3_para_dK_lKG_LF"
    folder_z_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\standard_design\standard_design"

    # Target speed for contact pressure analysis (default: 4500 RPM)
    target_speed = 4500

    # Run three-folder comparison with contact pressure analysis
    print("=" * 80)
    print("INTEGRATED POWER LOSS AND CONTACT PRESSURE ANALYSIS")
    print("=" * 80)
    print(f"Target speed for contact pressure analysis: {target_speed} RPM")
    print("Using T1 cumulative contact pressure analysis methods")
    print("=" * 80)

    main_three_folders(folder_x_path, folder_y_path, folder_z_path,
                       plot_type='relative', target_speed=target_speed)

    # ============================================================================
    # ALTERNATIVE CONFIGURATIONS
    # ============================================================================

    # Option 1: For TWO folders only (uncomment if needed)
    # main_two_folders(folder_x_path, folder_y_path, plot_type='relative', target_speed=4500)

    # Option 2: For CUSTOM folder configuration with different target speed
    # folder_paths = [
    #     ("Custom-Method-1", r"Z:\path\to\method1"),
    #     ("Custom-Method-2", r"Z:\path\to\method2"),
    #     ("Baseline", r"Z:\path\to\baseline"),
    # ]
    # main(folder_paths, plot_type='relative', target_speed=2000)

    # Option 3: For absolute value plots instead of relative percentages
    # main_three_folders(folder_x_path, folder_y_path, folder_z_path, plot_type='absolute', target_speed=4500)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Generated plots:")
    print("  1. Comparative power loss analysis (relative/absolute)")
    print("  2. Detailed breakdown (mechanical vs volumetric)")
    print("  3. Contact pressure vs total loss")
    print("  4. Dual-column comparison (Total Loss Bar | Contact Pressure vs Volumetric Loss Scatter)")
    print("  5. Contact pressure vs volumetric loss")
    print("  6. Contact pressure vs hydromechanical loss")
    print("  7. Contact pressure vs speed - BAR PLOT (linear scale only)")
    print("All plots saved as PNG files in the current directory")
    print("\nKEY FEATURES OF PLOT 7:")
    print("- Bar chart showing max cumulative contact pressure across all speeds")
    print("- Uses LINEAR scale for contact pressure (no log scale)")
    print("- Shows percentage differences from Standard-Design baseline")
    print("- Same bar chart format as your power loss vs speed plots")
    print("- Grouped bars for easy comparison between optimization methods")
    print("- Automatic baseline detection and difference calculation")
    print("\nTROUBLESHOOTING:")
    print("If plots are not displaying:")
    print("1. Check that contact pressure data exists for your simulation folders")
    print("2. Verify that T1 module imports are working correctly")
    print("3. Ensure matplotlib backend is properly configured")
    print("4. Check console output for detailed error messages")
    print("=" * 80)