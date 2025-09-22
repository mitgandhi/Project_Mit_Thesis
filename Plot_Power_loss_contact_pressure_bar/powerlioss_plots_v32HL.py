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


def read_operating_conditions(file_path):
    """
    Read operating conditions from operatingconditions.txt file
    Returns: (speed, beta, maxBeta, HP, LP)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        speed = None
        beta = None
        maxBeta = None
        HP = None
        LP = None

        for line in lines:
            line = line.strip()
            if line.startswith('speed') and 'betamax' not in line:
                # Extract speed value
                parts = line.split()
                if len(parts) >= 2:
                    speed = float(parts[1])
            elif line.startswith('beta') and 'max' not in line:
                # Extract beta value
                parts = line.split()
                if len(parts) >= 2:
                    beta = float(parts[1])
            elif line.startswith('betamax'):
                # Extract betamax value
                parts = line.split()
                if len(parts) >= 2:
                    maxBeta = float(parts[1])
            elif line.startswith('HP'):
                # Extract HP value
                parts = line.split()
                if len(parts) >= 2:
                    HP = float(parts[1])
            elif line.startswith('LP'):
                # Extract LP value
                parts = line.split()
                if len(parts) >= 2:
                    LP = float(parts[1])

        return speed, beta, maxBeta, HP, LP
    except Exception as e:
        print(f"Error reading operating conditions from {file_path}: {e}")
        return None, None, None, None, None


def calculate_efficiencies(data, displacement_volume=56):
    """
    Calculate volumetric, hydro-mechanical, and overall efficiencies
    Based on the MATLAB code methodology
    """
    efficiency_data = defaultdict(lambda: defaultdict(dict))

    for zeta in data.keys():
        for speed in data[zeta].keys():
            # Get the data for this condition
            condition_data = data[zeta][speed]

            # Extract values
            mech_loss = condition_data['mechanical_loss']  # W
            vol_loss = condition_data['volumetric_loss']  # W
            pressure = condition_data['pressure']  # bar
            displacement = condition_data['displacement']  # cc/rev

            # Use the speed from the folder structure
            speed_rpm = speed

            # Assume beta/maxBeta ratio (you can modify this based on your data)
            beta_norm = 1.0  # Assuming 100% displacement for now

            # Calculate theoretical flow (l/min)
            flow_theo = speed_rpm * displacement * beta_norm / 1000

            # Calculate pressure difference
            dp = pressure  # Assuming this is already the pressure difference

            # Estimate leakage from volumetric power loss
            # PlossL = leakage * dp / 600 * 1000 (W)
            # Therefore: leakage = PlossL * 600 / (dp * 1000) (l/min)
            if dp > 0:
                leakage = vol_loss * 600 / (dp * 1000)  # l/min
            else:
                leakage = 0

            # Net flow
            flow_net = flow_theo - leakage

            # Volumetric efficiency
            if flow_theo > 0:
                nu_vol = flow_net / flow_theo
            else:
                nu_vol = 0

            # Theoretical power (kW)
            P_theo = dp * flow_theo / 600

            # Total power loss (kW)
            Ploss_total = (mech_loss + vol_loss) / 1000

            # Actual power (kW)
            P_actual = P_theo - Ploss_total

            # Overall efficiency
            if P_theo > 0:
                nu_ges = P_actual / P_theo
            else:
                nu_ges = 0

            # Hydro-mechanical efficiency
            if nu_vol > 0:
                nu_hm = nu_ges / nu_vol
            else:
                nu_hm = 0

            # Store efficiency data
            efficiency_data[zeta][speed] = {
                'volumetric_eff': nu_vol * 100,  # %
                'hydromech_eff': nu_hm * 100,  # %
                'overall_eff': nu_ges * 100,  # %
                'flow_theo': flow_theo,  # l/min
                'leakage': leakage,  # l/min
                'flow_net': flow_net,  # l/min
                'power_theo': P_theo,  # kW
                'power_actual': P_actual,  # kW
                'total_loss': mech_loss + vol_loss  # W
            }

    return efficiency_data


def process_data_structure_new(root_path, zeta_value=5, pressure=320, displacement=100):
    """
    Process the new folder structure where speed folders are at the top level
    Structure: root_path/speed_folder/output/piston/piston.txt

    Parameters:
    - root_path: Path to the folder containing speed folders (e.g., 2000, 2500, etc.)
    - zeta_value: Fixed zeta/gamma value (default: 5)
    - pressure: Fixed pressure value (default: 320 bar)
    - displacement: Fixed displacement value (default: 100%)

    Returns: Dictionary with zeta values as keys and speed data as values
    """
    data = defaultdict(lambda: defaultdict(dict))

    print(f"Scanning directory: {root_path}")

    # Check if the root path exists
    if not os.path.exists(root_path):
        print(f"Error: Root path does not exist: {root_path}")
        return data

    # Iterate through all items in the root directory
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)

        # Check if it's a directory and if the name represents a speed (numeric)
        if os.path.isdir(item_path) and item.isdigit():
            speed = int(item)
            print(f"Processing speed folder: {speed}")

            # Look for piston.txt file in the expected location
            piston_file_paths = [
                os.path.join(item_path, 'output', 'piston', 'piston.txt'),  # Standard path
                os.path.join(item_path, 'piston', 'piston.txt'),  # Alternative path
                os.path.join(item_path, 'piston.txt')  # Direct path
            ]

            piston_file = None
            for path in piston_file_paths:
                if os.path.exists(path):
                    piston_file = path
                    break

            if piston_file:
                print(f"  Found piston file: {piston_file}")
                mech_loss, vol_loss = read_piston_data(piston_file)

                if mech_loss is not None and vol_loss is not None:
                    # Store data with the fixed zeta value
                    data[zeta_value][speed] = {
                        'mechanical_loss': mech_loss,
                        'volumetric_loss': vol_loss,
                        'pressure': pressure,
                        'displacement': displacement,
                        'total_loss': mech_loss + vol_loss
                    }
                    print(f"  Successfully processed: Mech={mech_loss:.2f}W, Vol={vol_loss:.2f}W")
                else:
                    print(f"  Warning: Could not extract power loss data from {piston_file}")
            else:
                print(f"  Warning: No piston.txt file found in {item_path}")
                print(f"    Searched in:")
                for path in piston_file_paths:
                    print(f"      {path}")

    if not data:
        print("No data was processed. Please check:")
        print("1. The root path is correct")
        print("2. Speed folders (numeric names) exist")
        print("3. piston.txt files exist in the expected locations")
        print("4. piston.txt files contain the required columns")
    else:
        print(f"\nSuccessfully processed {len(data[zeta_value])} speed conditions")

    return data


def create_total_power_loss_plot(data, pressure_val=None, displacement_val=None, relative=True):
    """
    Create a stacked bar chart for power losses
    X-axis: Operating Conditions (speeds)
    Y-axis: Power loss in % or W
    Each bar shows volumetric loss (light) at bottom and mechanical loss (dark) on top
    """
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

    # Extract and sort speed and zeta values
    all_speeds = set()
    for zeta_data in data.values():
        all_speeds.update(zeta_data.keys())
    speeds = sorted(list(all_speeds))
    zeta_values = sorted(data.keys())

    # Prepare data structures for volumetric and mechanical losses separately
    vol_loss_data = {}
    mech_loss_data = {}

    for speed in speeds:
        vol_loss_data[speed] = {}
        mech_loss_data[speed] = {}
        for zeta in zeta_values:
            if speed in data[zeta]:
                vol_loss_data[speed][zeta] = data[zeta][speed]['volumetric_loss']
                mech_loss_data[speed][zeta] = data[zeta][speed]['mechanical_loss']
            else:
                vol_loss_data[speed][zeta] = 0
                mech_loss_data[speed][zeta] = 0

    # Calculate relative percentages if needed
    if relative:
        # Calculate grand total of all power losses
        grand_total = 0
        for speed in speeds:
            for zeta in zeta_values:
                grand_total += vol_loss_data[speed][zeta] + mech_loss_data[speed][zeta]

        # Convert to percentages
        if grand_total > 0:
            for speed in speeds:
                for zeta in zeta_values:
                    vol_loss_data[speed][zeta] = (vol_loss_data[speed][zeta] / grand_total) * 100
                    mech_loss_data[speed][zeta] = (mech_loss_data[speed][zeta] / grand_total) * 100

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define base colors for different zeta values
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Set up bar positions
    x = np.arange(len(speeds))
    width = 0.8 / len(zeta_values)  # Width of individual bars

    # Create stacked bars for each zeta value
    for i, zeta in enumerate(zeta_values):
        # Get volumetric and mechanical loss values for this zeta
        vol_values = [vol_loss_data[speed][zeta] for speed in speeds]
        mech_values = [mech_loss_data[speed][zeta] for speed in speeds]

        # Position for this zeta group
        pos = x + (i - len(zeta_values) / 2 + 0.5) * width

        # Get base color for this zeta
        base_color = base_colors[i % len(base_colors)]

        # Create light version for volumetric loss (bottom part)
        from matplotlib.colors import to_rgba
        light_color = list(to_rgba(base_color))
        light_color[3] = 0.4  # Make it more transparent/lighter

        # Create dark version for mechanical loss (top part)
        dark_color = list(to_rgba(base_color))
        dark_color[3] = 0.9  # Make it more opaque/darker

        # Bottom bars (volumetric loss) - light color
        vol_bars = ax.bar(pos, vol_values, width,
                          label=f'Volumetric Loss - Zeta {zeta}°',
                          color=light_color,
                          edgecolor='white', linewidth=0.5)

        # Top bars (mechanical loss) - dark color, stacked on volumetric
        mech_bars = ax.bar(pos, mech_values, width, bottom=vol_values,
                           label=f'Mechanical Loss - Zeta {zeta}°',
                           color=dark_color,
                           edgecolor='white', linewidth=0.5)

        # # Add value labels on bars
        # for j, (vol_val, mech_val) in enumerate(zip(vol_values, mech_values)):
        #     total_val = vol_val + mech_val
        #     if total_val > 0:
        #         # Label for total value at the top
        #         if relative:
        #             ax.text(pos[j], total_val + (max(vol_values + mech_values) * 0.01),
        #                     f'{total_val:.1f}%', ha='center', va='bottom',
        #                     fontweight='bold', fontsize=9)
        #         else:
        #             ax.text(pos[j], total_val + (max(vol_values + mech_values) * 0.01),
        #                     f'{total_val:.0f}W', ha='center', va='bottom',
        #                     fontweight='bold', fontsize=9)

    # Formatting with bold axis labels
    ax.set_xlabel('Operating Conditions (Speed)', fontweight='bold')
    if relative:
        ax.set_ylabel('Power Loss (%)', fontweight='bold')
        # Set y-limit with space for labels
        max_val = max([vol_loss_data[speed][zeta] + mech_loss_data[speed][zeta]
                       for speed in speeds for zeta in zeta_values])
        ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 100)
    else:
        ax.set_ylabel('Power Loss (W)', fontweight='bold')
        max_val = max([vol_loss_data[speed][zeta] + mech_loss_data[speed][zeta]
                       for speed in speeds for zeta in zeta_values])
        ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1)

    # Set x-axis labels
    ax.set_xticks(x)
    speed_labels = [f'OC{i + 1}\n({speed} RPM)' for i, speed in enumerate(speeds)]
    ax.set_xticklabels(speed_labels)

    # Title with pressure and displacement info
    title_prefix = "Relative " if relative else ""
    ax.set_title(
        f'Pressure: {pressure_val} bar, Displacement: {displacement_val} %,  Gamma: {zeta}°')

    # Create custom legend to show volumetric (light) and mechanical (dark) losses
    legend_elements = []
    for i, zeta in enumerate(zeta_values):
        base_color = base_colors[i % len(base_colors)]

        # Light patch for volumetric
        from matplotlib.patches import Patch
        light_color = list(to_rgba(base_color))
        light_color[3] = 0.4
        legend_elements.append(Patch(facecolor=light_color, edgecolor='white',
                                     label=f'Volumetric loss'))

        # Dark patch for mechanical
        dark_color = list(to_rgba(base_color))
        dark_color[3] = 0.9
        legend_elements.append(Patch(facecolor=dark_color, edgecolor='white',
                                     label=f'Mechanical loss'))

    # Legend below x-axis
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25),
              ncol=4, frameon=True, fancybox=True, shadow=False, framealpha=0.9,
              edgecolor='black', fontsize=16, markerscale=1.2, columnspacing=1.5,
              handlelength=2.0, handletextpad=0.5, borderpad=1.0)

    # Grid
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the figure
    fig.savefig('power_loss_analysis_stacked__v32.png', dpi=300, bbox_inches='tight')

    # Return the figure and axis objects
    return fig, ax


def main(root_path, plot_type='relative', zeta_value=5, pressure=320, displacement=100):
    """
    Main function to process data and create visualization for new folder structure

    Parameters:
    - root_path: Path to folder containing speed folders
    - plot_type: 'relative' for normalized percentage plots, 'absolute' for actual values
    - zeta_value: Fixed zeta/gamma value (default: 5)
    - pressure: Fixed pressure value (default: 320 bar)
    - displacement: Fixed displacement percentage (default: 100%)
    """
    print("Processing new data structure...")
    print(f"Root path: {root_path}")
    print(f"Fixed parameters: Zeta/Gamma={zeta_value}°, Pressure={pressure} bar, Displacement={displacement}%")

    data = process_data_structure_new(root_path, zeta_value, pressure, displacement)

    if not data:
        print("No data found. Please check the folder structure and file paths.")
        return

    print(f"Found data for {len(data)} zeta values")
    print(f"Zeta values: {sorted(data.keys())}")

    # Print found speeds for verification
    for zeta in data:
        speeds = sorted(data[zeta].keys())
        print(f"Zeta {zeta}: {len(speeds)} speeds found: {speeds}")

    # Calculate efficiencies (using displacement in cc/rev - convert percentage to actual displacement)
    displacement_cc = displacement * 56 / 100  # Assuming base displacement of 56 cc/rev for V60N
    print(f"\nCalculating efficiencies with displacement: {displacement_cc} cc/rev...")
    efficiency_data = calculate_efficiencies(data, displacement_cc)

    # Determine if we want relative (normalized) plots
    relative = (plot_type.lower() == 'relative')

    # Create and show the total power loss plot
    if relative:
        print("\nCreating Relative (Normalized) Total Power Loss Plot...")
    else:
        print("\nCreating Absolute Total Power Loss Plot...")

    try:
        fig1, ax1 = create_total_power_loss_plot(data, pressure, displacement, relative=relative)
        if fig1 is not None and ax1 is not None:
            plt.figure(fig1.number)
            plt.show()
            print("Power loss plot displayed successfully")
        else:
            print("?? Could not create power loss plot")
    except Exception as e:
        print(f"?? Error creating power loss plot: {e}")
        import traceback
        traceback.print_exc()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\n=== POWER LOSSES ===")
    for zeta in sorted(data.keys()):
        print(f"\nGamma/Zeta {zeta}°:")
        for speed in sorted(data[zeta].keys()):
            mech = data[zeta][speed]['mechanical_loss']
            vol = data[zeta][speed]['volumetric_loss']
            total = data[zeta][speed]['total_loss']
            print(f"  Speed {speed} RPM: Mech={mech:.2f}W, Vol={vol:.2f}W, Total={total:.2f}W")


if __name__ == "__main__":
    # Update this path to your actual data folder
    root_folder_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T05_variable speed_ speedK_1"

    # Parameters for your specific case
    zeta_gamma_value = 5  # Fixed gamma/zeta value
    pressure_value = 320  # Fixed pressure in bar
    displacement_percent = 100  # Fixed displacement in %

    # For absolute value plots:
    main(root_folder_path,
         plot_type='absolute',
         zeta_value=zeta_gamma_value,
         pressure=pressure_value,
         displacement=displacement_percent)

    # Uncomment below for relative percentage plots:
    # main(root_folder_path,
    #      plot_type='relative',
    #      zeta_value=zeta_gamma_value,
    #      pressure=pressure_value,
    #      displacement=displacement_percent)