import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict


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


def process_data_structure(root_path):
    """
    Process the entire folder structure to extract data
    Returns: Dictionary with zeta values as keys and speed data as values
    """
    data = defaultdict(lambda: defaultdict(dict))

    # Iterate through zeta folders
    for item in os.listdir(root_path):
        if item.startswith('zeta'):
            zeta_value = int(item.replace('zeta', ''))
            zeta_path = os.path.join(root_path, item)

            if os.path.isdir(zeta_path):
                # Look for V60N_S4_dp_* folders
                for subfolder in os.listdir(zeta_path):
                    parsed = parse_folder_name(subfolder)
                    if parsed:
                        pump_series, series_num, pressure, speed, displacement = parsed

                        # Path to piston.txt file
                        piston_file = os.path.join(zeta_path, subfolder, 'output', 'piston', 'piston.txt')

                        if os.path.exists(piston_file):
                            mech_loss, vol_loss = read_piston_data(piston_file)
                            if mech_loss is not None and vol_loss is not None:
                                data[zeta_value][speed] = {
                                    'mechanical_loss': mech_loss,
                                    'volumetric_loss': vol_loss,
                                    'pressure': pressure,
                                    'displacement': displacement,
                                    'total_loss': mech_loss + vol_loss
                                }

    return data


def create_total_power_loss_plot(data, pressure_val=None, displacement_val=None, relative=True):
    """
    Create a stacked bar chart for total power losses (mechanical + volumetric combined)
    X-axis: Zeta values (Piston inclination angle), Colored segments: Operating Conditions (speeds)
    Y-axis: Power loss in %
    """
    # Extract unique speed values and sort them
    all_speeds = set()
    for zeta_data in data.values():
        all_speeds.update(zeta_data.keys())
    speeds = sorted(list(all_speeds))

    # Extract zeta values and sort them
    zeta_values = sorted(data.keys())

    # Prepare total power loss data for plotting
    # Structure: [zeta][speed] = total_loss_value
    total_losses = []

    for zeta in zeta_values:
        total_for_zeta = []
        for speed in speeds:
            if speed in data[zeta]:
                # Combine mechanical and volumetric losses
                total_loss = data[zeta][speed]['mechanical_loss'] + data[zeta][speed]['volumetric_loss']
                total_for_zeta.append(total_loss)
            else:
                total_for_zeta.append(0)
        total_losses.append(total_for_zeta)

    # Convert to numpy array for easier manipulation
    total_losses = np.array(total_losses)

    # Normalize data to percentages if relative is True
    if relative:
        # Calculate totals for each zeta
        totals = np.sum(total_losses, axis=1)
        # Avoid division by zero
        totals[totals == 0] = 1
        # Convert to percentages
        total_losses_norm = (total_losses.T / totals * 100).T
    else:
        total_losses_norm = total_losses

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for different operating conditions (speeds)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    x = np.arange(len(zeta_values))
    width = 0.6

    # Create stacked bars for total losses
    bottom = np.zeros(len(zeta_values))

    for i, speed in enumerate(speeds):
        color = colors[i % len(colors)]
        values = total_losses_norm[:, i]
        ax.bar(x, values, width, bottom=bottom,
               label=f'OC{i + 1} ({speed} RPM)', color=color, alpha=0.8)

        # Add percentage labels on bars if relative
        if relative:
            for j, val in enumerate(values):
                if val > 2:  # Only show labels for values > 2%
                    ax.text(j, bottom[j] + val / 2, f'{val:.1f}%',
                            ha='center', va='center', fontweight='bold', fontsize=9)

        bottom += values

    # Formatting
    ax.set_xlabel('Piston Inclination Angle (°)')
    if relative:
        ax.set_ylabel('Power Loss (%)')
        ax.set_ylim(0, 100)
    else:
        ax.set_ylabel('Power Loss (W)')

    ax.set_xticks(x)
    # Show only zeta numbers on x-axis
    ax.set_xticklabels([f'{zeta}' for zeta in zeta_values])

    # Title with pressure and displacement info
    title_prefix = "Relative " if relative else ""
    if pressure_val and displacement_val:
        ax.set_title(
            f'Pressure: {pressure_val} bar, Displacement: {displacement_val} cc/rev')
    else:
        ax.set_title(f'{title_prefix}Total Power Loss Analysis')

    # Legend
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def main(root_path, plot_type='relative'):
    """
    Main function to process data and create visualization
    plot_type: 'relative' for normalized percentage plots, 'absolute' for actual values
    """
    print("Processing data structure...")
    data = process_data_structure(root_path)

    if not data:
        print("No data found. Please check the folder structure and file paths.")
        return

    # Get pressure and displacement values for title (from first available data point)
    pressure_val = None
    displacement_val = None
    for zeta_data in data.values():
        for speed_data in zeta_data.values():
            pressure_val = speed_data['pressure']
            displacement_val = speed_data['displacement']
            break
        if pressure_val:
            break

    print(f"Found data for {len(data)} zeta values")
    print(f"Zeta values: {sorted(data.keys())}")

    # Determine if we want relative (normalized) plots
    relative = (plot_type.lower() == 'relative')

    # Create and show the total power loss plot
    if relative:
        print("\nCreating Relative (Normalized) Total Power Loss Plot...")
    else:
        print("\nCreating Absolute Total Power Loss Plot...")

    fig, ax = create_total_power_loss_plot(data, pressure_val, displacement_val, relative=relative)
    plt.figure(fig.number)
    plt.savefig(root_folder_path + ".png", transparent = True)
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    for zeta in sorted(data.keys()):
        print(f"\nZeta {zeta}:")
        for speed in sorted(data[zeta].keys()):
            mech = data[zeta][speed]['mechanical_loss']
            vol = data[zeta][speed]['volumetric_loss']
            total = data[zeta][speed]['total_loss']
            print(f"  Speed {speed}: Mech={mech:.2f}W, Vol={vol:.2f}W, Total={total:.2f}W")

    # Print normalized statistics if relative plots
    if relative:
        print("\nNormalized Statistics (Percentages):")
        for zeta in sorted(data.keys()):
            print(f"\nZeta {zeta}:")

            # Calculate totals for this zeta across all speeds
            total_combined = sum(data[zeta][speed]['total_loss'] for speed in data[zeta].keys())

            # Calculate percentages for each speed
            for speed in sorted(data[zeta].keys()):
                total_pct = (data[zeta][speed]['total_loss'] / total_combined * 100) if total_combined > 0 else 0
                print(f"  Speed {speed}: Total={total_pct:.1f}%")




# Example usage:
if __name__ == "__main__":
    # Replace this with the actual path to your data folder
    root_folder_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run3_V60N_with_different_inclination"

    # # For relative (normalized percentage) plots:
    # main(root_folder_path, plot_type='relative')

    # For absolute value plots:
    main(root_folder_path, plot_type='absolute')
    print(
        "Please update the root_folder_path variable with your actual data folder path and uncomment the main() call.")