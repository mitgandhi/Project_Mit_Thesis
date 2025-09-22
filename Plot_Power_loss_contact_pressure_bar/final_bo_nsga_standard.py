import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# Ensure matplotlib backend is set
import matplotlib

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

matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer


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


def process_multiple_folder_structure(folder_paths):
    """
    Process multiple folders each containing speed subfolders (2000, 2500, etc.)
    Structure: folder_x/2000/output/piston/piston.txt
               folder_y/2000/output/piston/piston.txt
               folder_z/2000/output/piston/piston.txt

    Args:
        folder_paths: List of tuples [(folder_name, folder_path), ...]

    Returns: Dictionary with structure {folder_name: {speed: data}}
    """
    all_data = {}

    # Process all folders
    folders_to_process = folder_paths

    for folder_name, folder_path in folders_to_process:
        print(f"\n=== Processing {folder_name}: {folder_path} ===")

        if not os.path.exists(folder_path):
            print(f"Error: Path does not exist: {folder_path}")
            continue

        folder_data = {}

        # List all items in folder
        folder_items = os.listdir(folder_path)
        print(f"Found items in {folder_name}: {folder_items}")

        # Look for speed folders (folders with numeric names like 2000, 2500, etc.)
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

            # Path to piston.txt file (directly in speed folder)
            piston_file = os.path.join(speed_folder_path, 'output', 'piston', 'piston.txt')

            if os.path.exists(piston_file):
                print(f"    Found piston file: {piston_file}")
                mech_loss, vol_loss = read_piston_data(piston_file)
                if mech_loss is not None and vol_loss is not None:
                    folder_data[speed_val] = {
                        'mechanical_loss': mech_loss,
                        'volumetric_loss': vol_loss,
                        'total_loss': mech_loss + vol_loss,
                        'speed': speed_val
                    }
                    print(
                        f"    Successfully processed: mech={mech_loss:.2f}W, vol={vol_loss:.2f}W, total={mech_loss + vol_loss:.2f}W")
                else:
                    print(f"    Failed to read piston data")
            else:
                print(f"    Piston file not found: {piston_file}")

        all_data[folder_name] = folder_data

    return all_data


def create_comparative_power_loss_plot(all_data, relative=True):
    """
    Create a comparative plot showing power losses from multiple folders
    X-axis: Speed values (2000, 2500, etc.)
    Grouped bars: OPTIMAL-BAYESIAN vs OPTIMAL-NSGA vs BASE_MODEL
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
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

    # Colors for different folders - using distinct, professional colors
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
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

    # Convert to relative percentages if requested
    if relative:
        total_sum = sum(sum(data) for data in all_folder_data.values())
        if total_sum > 0:
            for folder_name in folder_names:
                all_folder_data[folder_name] = [(val / total_sum) * 100 for val in all_folder_data[folder_name]]

    # Create bars for each folder
    for i, folder_name in enumerate(folder_names):
        folder_data = all_folder_data[folder_name]

        # Position for this folder's bars
        pos = x + (i - len(folder_names) / 2 + 0.5) * width

        bars = ax.bar(pos, folder_data, width,
                      label=folder_name, color=folder_colors[folder_name], alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, folder_data):
            if val > 0:
                if relative and val > 1:  # Only show labels for values > 1%
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{val:.1f}%', ha='center', va='bottom',
                            fontweight='bold', fontsize=9, rotation=0)
                elif not relative:
                    max_val = max(max(data) for data in all_folder_data.values())
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.01,
                            f'{val:.0f}W', ha='center', va='bottom',
                            fontweight='bold', fontsize=9, rotation=0)

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
    ax.set_title(f'{title_prefix}Power Loss Comparison: Optimization Methods',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()

    # Save the figure
    fig.savefig('comparative_power_loss_analysis.png', dpi=300, bbox_inches='tight')

    return fig, ax


def create_detailed_breakdown_plot(all_data):
    """
    Create a single grouped bar plot showing mechanical vs volumetric losses for all optimization methods
    Similar to the first plot but with mechanical and volumetric breakdown instead of total losses
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

    print(f"Speeds: {speeds}")
    print(f"Folders: {folder_names}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Colors for different folders - using distinct, professional colors
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    folder_colors = {folder: colors[i % len(colors)] for i, folder in enumerate(folder_names)}

    # Prepare data
    x = np.arange(len(speeds))
    width = 0.8 / (len(folder_names) * 2)  # Divide by 2 because we have mechanical AND volumetric bars

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

    # Create grouped bars for each folder (mechanical and volumetric side by side)
    bar_positions = {}
    current_pos = 0

    for i, folder_name in enumerate(folder_names):
        mech_data = all_folder_mech_data[folder_name]
        vol_data = all_folder_vol_data[folder_name]

        # Position for mechanical bars
        mech_pos = x + current_pos * width
        # Position for volumetric bars (right next to mechanical)
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

        # Update position for next folder (skip 2 positions: mechanical + volumetric + gap)
        current_pos += 2.5

    # Formatting
    ax.set_xlabel('Speed (RPM)', fontweight='bold')
    ax.set_ylabel('Power Loss (W)', fontweight='bold')
    ax.set_title('Mechanical vs Volumetric Power Loss Comparison', fontweight='bold', fontsize=14)

    # Adjust x-tick positions to be centered between the grouped bars
    tick_offset = (len(folder_names) * 2.5 - 0.5) * width / 2
    ax.set_xticks(x + tick_offset)
    ax.set_xticklabels([str(speed) for speed in speeds])

    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save the figure
    fig.savefig('detailed_breakdown_analysis.png', dpi=300, bbox_inches='tight')

    return fig, ax


def create_summary_comparison_table(all_data):
    """
    Create a summary table comparing multiple folders
    """
    print("\n" + "=" * 120)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 120)

    # Get all speeds and folder names
    all_speeds = set()
    for folder_data in all_data.values():
        all_speeds.update(folder_data.keys())

    speeds = sorted(list(all_speeds))
    folder_names = list(all_data.keys())

    # Create header
    header = f"{'Speed (RPM)':<12}"
    for folder_name in folder_names:
        header += f" {folder_name + ' (W)':<20}"

    # Add difference columns for comparison with first folder
    if len(folder_names) > 1:
        base_folder = folder_names[0]
        for i, folder_name in enumerate(folder_names[1:], 1):
            header += f" {'Diff vs ' + base_folder + ' (W)':<25}"
            header += f" {'% Diff vs ' + base_folder:<20}"

    print(header)
    print("-" * len(header))

    for speed in speeds:
        # Get data for all folders
        folder_totals = {}
        for folder_name in folder_names:
            if speed in all_data.get(folder_name, {}):
                folder_totals[folder_name] = all_data[folder_name][speed]['total_loss']
            else:
                folder_totals[folder_name] = 0

        # Build row
        row = f"{speed:<12}"
        for folder_name in folder_names:
            row += f" {folder_totals[folder_name]:<20.2f}"

        # Add differences (compared to first folder)
        if len(folder_names) > 1:
            base_value = folder_totals[folder_names[0]]
            for folder_name in folder_names[1:]:
                current_value = folder_totals[folder_name]
                difference = current_value - base_value

                if base_value > 0:
                    percent_diff = (difference / base_value) * 100
                else:
                    percent_diff = 0 if current_value == 0 else float('inf')

                row += f" {difference:<25.2f}"
                row += f" {percent_diff:<20.1f}"

        print(row)

    # Print detailed breakdown
    print("\n" + "=" * 120)
    print("DETAILED BREAKDOWN")
    print("=" * 120)

    for folder_name, folder_data in all_data.items():
        print(f"\n--- {folder_name} ---")
        print(f"{'Speed (RPM)':<12} {'Mechanical (W)':<15} {'Volumetric (W)':<15} {'Total (W)':<15}")
        print("-" * 60)

        for speed in sorted(folder_data.keys()):
            data = folder_data[speed]
            print(
                f"{speed:<12} {data['mechanical_loss']:<15.2f} {data['volumetric_loss']:<15.2f} {data['total_loss']:<15.2f}")


def main(folder_paths, plot_type='relative'):
    """
    Main function to process multiple folders and create comparative visualization

    Args:
        folder_paths: List of tuples [(folder_name, folder_path), ...]
                     e.g., [("OPTIMAL-BAYESIAN", "/path/to/x"), ("OPTIMAL-NSGA", "/path/to/y"), ("BASE_MODEL", "/path/to/z")]
    """
    print("Processing multiple-folder data structure...")
    all_data = process_multiple_folder_structure(folder_paths)

    if not all_data or not any(all_data.values()):
        print("No data found. Please check the folder structure and file paths.")
        print("Expected structure:")
        print("  folder_x/[speed_folders]/output/piston/piston.txt")
        print("  folder_y/[speed_folders]/output/piston/piston.txt")
        print("  folder_z/[speed_folders]/output/piston/piston.txt")
        return

    print(f"\nFound data for folders: {list(all_data.keys())}")

    # Print detailed summary
    create_summary_comparison_table(all_data)

    # Determine if we want relative (normalized) plots
    relative = (plot_type.lower() == 'relative')

    # Create and show the comparative plot
    if relative:
        print("\nCreating Relative (Normalized) Comparative Power Loss Plot...")
    else:
        print("\nCreating Absolute Comparative Power Loss Plot...")

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

    # Create detailed breakdown plot
    print("\nCreating Detailed Breakdown Plot...")
    try:
        fig2, axes2 = create_detailed_breakdown_plot(all_data)
        if fig2 is not None and axes2 is not None:
            plt.figure(fig2.number)
            plt.show()
            print("Detailed breakdown plot displayed successfully")
        else:
            print("⚠️ Could not create detailed breakdown plot")
    except Exception as e:
        print(f"⚠️ Error creating detailed breakdown plot: {e}")


# Convenience functions with updated labels
def main_two_folders(folder_x_path, folder_y_path, plot_type='relative'):
    """Convenience function for two folders with proper labels"""
    folder_paths = [
        ("OPTIMAL-BAYESIAN", folder_x_path),
        ("OPTIMAL-NSGA", folder_y_path)
    ]
    main(folder_paths, plot_type)


def main_three_folders(folder_x_path, folder_y_path, folder_z_path, plot_type='relative'):
    """Convenience function for three folders with proper labels"""
    folder_paths = [
        ("Optimal-Bayesian", folder_x_path),
        ("Optimal-NSGA-III", folder_y_path),
        ("Standard-Design", folder_z_path)
    ]
    main(folder_paths, plot_type)


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION - Update these paths to your actual folders
    # ============================================================================

    # Option 1: For THREE folders with proper labels
    folder_x_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\standard_design\optimal_BO_3_para_dK_lKG_LF"
    folder_y_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\standard_design\optimal_nsga_3_para_dK_lKG_LF"
    folder_z_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\standard_design\standard_design"

    # Run three-folder comparison with proper labels
    main_three_folders(folder_x_path, folder_y_path, folder_z_path, plot_type='relative')

    # ============================================================================

    # Option 2: For TWO folders with proper labels (uncomment if you only have two folders)
    # main_two_folders(folder_x_path, folder_y_path, plot_type='relative')

    # ============================================================================

    # Option 3: For CUSTOM folder configuration with explicit labels
    # folder_paths = [
    #     ("OPTIMAL-BAYESIAN", r"Z:\path\to\optimal_BO_folder"),
    #     ("OPTIMAL-NSGA", r"Z:\path\to\optimal_nsga_folder"),
    #     ("BASE_MODEL", r"Z:\path\to\standard_design_folder"),
    # ]
    # main(folder_paths, plot_type='relative')

    # ============================================================================

    # For absolute value plots instead of relative percentages:
    # main_three_folders(folder_x_path, folder_y_path, folder_z_path, plot_type='absolute')

    print("\nAnalysis complete! The plots now use proper labels:")
    print("  - X folder → OPTIMAL-BAYESIAN")
    print("  - Y folder → OPTIMAL-NSGA")
    print("  - Z folder → BASE_MODEL")