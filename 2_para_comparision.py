import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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


def process_design_paths_data(design_paths, pressure=320, displacement=100):
    """
    Process multiple design folders with custom paths

    Parameters:
    - design_paths: Dictionary with design names as keys and folder paths as values
                   e.g., {'industrial': 'path/to/industrial', 'optimal': 'path/to/optimal'}
    - pressure: Fixed pressure value (default: 320 bar)
    - displacement: Fixed displacement value (default: 100%)

    Returns: Dictionary with design types as keys and speed data as values
    """
    data = defaultdict(lambda: defaultdict(dict))

    for design_name, design_path in design_paths.items():
        print(f"\nProcessing {design_name} design from: {design_path}")

        # Check if the design path exists
        if not os.path.exists(design_path):
            print(f"Error: Design path does not exist: {design_path}")
            continue

        # Iterate through all items in the design directory to find speed folders
        for item in os.listdir(design_path):
            item_path = os.path.join(design_path, item)

            # Check if it's a directory and if the name represents a speed (numeric and >= 0)
            if os.path.isdir(item_path):
                try:
                    speed = int(item)
                    if speed < 0:  # Skip negative speeds
                        print(f"    Skipping negative speed folder: {speed}")
                        continue
                except ValueError:
                    # Skip non-numeric folders
                    continue
                print(f"  Processing speed folder: {speed}")

                # Look for piston.txt file in the expected locations
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
                    print(f"    Found piston file: {piston_file}")
                    mech_loss, vol_loss = read_piston_data(piston_file)

                    if mech_loss is not None and vol_loss is not None:
                        data[design_name][speed] = {
                            'mechanical_loss': mech_loss,
                            'volumetric_loss': vol_loss,
                            'total_loss': mech_loss + vol_loss,
                            'pressure': pressure,
                            'displacement': displacement
                        }
                        print(
                            f"    Successfully processed: Mech={mech_loss:.2f}W, Vol={vol_loss:.2f}W, Total={mech_loss + vol_loss:.2f}W")
                    else:
                        print(f"    Warning: Could not extract power loss data from {piston_file}")
                else:
                    print(f"    Warning: No piston.txt file found in {item_path}")

    if not data:
        print("No data was processed. Please check:")
        print("1. The design paths are correct")
        print("2. Speed folders (numeric names) exist")
        print("3. piston.txt files exist in the expected locations")
        print("4. piston.txt files contain the required columns")
    else:
        total_conditions = sum(len(speeds) for speeds in data.values())
        print(f"\nSuccessfully processed {len(data)} designs with {total_conditions} total speed conditions")

    return data


def create_multi_speed_comparison_plot(data, pressure_val=None, displacement_val=None, relative=False):
    """
    Create a stacked bar chart comparing designs across multiple speeds
    X-axis: Speed conditions
    Y-axis: Power loss in W or %
    Each design gets its own color, with volumetric (light) and mechanical (dark) losses stacked
    """
    if not data:
        print("No data available for plotting")
        return None, None

    # Extract all unique speeds across all designs
    all_speeds = set()
    for design_data in data.values():
        all_speeds.update(design_data.keys())
    speeds = sorted(list(all_speeds))

    if not speeds:
        print("No speed data found")
        return None, None

    # Extract design names
    design_names = list(data.keys())
    n_designs = len(design_names)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for different designs
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Set up bar positions
    x = np.arange(len(speeds))
    width = 0.8 / n_designs  # Width of individual bars

    # Prepare data for relative plotting if needed
    if relative:
        # Calculate grand total of all power losses
        grand_total = 0
        for design_name in design_names:
            for speed in speeds:
                if speed in data[design_name]:
                    grand_total += data[design_name][speed]['total_loss']

    # Create bars for each design
    for i, design_name in enumerate(design_names):
        # Prepare data arrays for this design
        vol_losses = []
        mech_losses = []

        for speed in speeds:
            if speed in data[design_name]:
                vol_loss = data[design_name][speed]['volumetric_loss']
                mech_loss = data[design_name][speed]['mechanical_loss']

                if relative and grand_total > 0:
                    vol_loss = (vol_loss / grand_total) * 100
                    mech_loss = (mech_loss / grand_total) * 100

                vol_losses.append(vol_loss)
                mech_losses.append(mech_loss)
            else:
                vol_losses.append(0)
                mech_losses.append(0)

        # Position for this design's bars
        pos = x + (i - n_designs / 2 + 0.5) * width

        # Get colors for this design
        base_color = base_colors[i % len(base_colors)]

        # Create light version for volumetric loss (bottom part)
        from matplotlib.colors import to_rgba
        light_color = list(to_rgba(base_color))
        light_color[3] = 0.5  # Make it more transparent/lighter

        # Create dark version for mechanical loss (top part)
        dark_color = list(to_rgba(base_color))
        dark_color[3] = 0.9  # Make it more opaque/darker

        # Bottom bars (volumetric loss) - light color
        vol_bars = ax.bar(pos, vol_losses, width,
                          label=f'Volumetric Loss - {design_name.capitalize()}',
                          color=light_color,
                          edgecolor='white', linewidth=0.5)

        # Top bars (mechanical loss) - dark color, stacked on volumetric
        mech_bars = ax.bar(pos, mech_losses, width, bottom=vol_losses,
                           label=f'Mechanical Loss - {design_name.capitalize()}',
                           color=dark_color,
                           edgecolor='white', linewidth=0.5)

        # Add value labels on bars for non-zero values
        for j, (vol_val, mech_val) in enumerate(zip(vol_losses, mech_losses)):
            total_val = vol_val + mech_val
            if total_val > 0:
                if relative:
                    ax.text(pos[j], total_val + (max(vol_losses + mech_losses) * 0.01),
                            f'{total_val:.1f}%', ha='center', va='bottom',
                            fontweight='bold', fontsize=9)
                else:
                    ax.text(pos[j], total_val + (max(vol_losses + mech_losses) * 0.01),
                            f'{total_val:.0f}W', ha='center', va='bottom',
                            fontweight='bold', fontsize=9)

    # Formatting
    ax.set_xlabel('Operating Conditions (Speed)', fontweight='bold')
    if relative:
        ax.set_ylabel('Power Loss (%)', fontweight='bold')
    else:
        ax.set_ylabel('Power Loss (W)', fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(x)
    speed_labels = [f'OC{i + 1}\n({speed} RPM)' for i, speed in enumerate(speeds)]
    ax.set_xticklabels(speed_labels)

    # Title with parameters
    title_parts = []
    if pressure_val:
        title_parts.append(f'Pressure: {pressure_val} bar')
    if displacement_val:
        title_parts.append(f'Displacement: {displacement_val}%')

    title = 'Power Loss Comparison: ' + ' vs '.join([name.capitalize() for name in design_names])
    if title_parts:
        title += '\n' + ', '.join(title_parts)
    ax.set_title(title, fontweight='bold', pad=20)

    # Create custom legend
    legend_elements = []
    for i, design_name in enumerate(design_names):
        base_color = base_colors[i % len(base_colors)]

        # Light patch for volumetric
        from matplotlib.patches import Patch
        light_color = list(to_rgba(base_color))
        light_color[3] = 0.5
        legend_elements.append(Patch(facecolor=light_color, edgecolor='white',
                                     label=f'Volumetric - {design_name.capitalize()}'))

        # Dark patch for mechanical
        dark_color = list(to_rgba(base_color))
        dark_color[3] = 0.9
        legend_elements.append(Patch(facecolor=dark_color, edgecolor='white',
                                     label=f'Mechanical - {design_name.capitalize()}'))

    # Legend below x-axis
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=min(4, len(legend_elements)), frameon=True, fancybox=True, shadow=False,
              framealpha=0.9, edgecolor='black', fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plot_type = "relative" if relative else "absolute"
    fig.savefig(f'multi_design_comparison_{plot_type}.png', dpi=300, bbox_inches='tight')

    return fig, ax


def print_detailed_comparison(data):
    """Print detailed comparison across all speeds and designs"""
    if not data:
        print("No data available for comparison")
        return

    print("\n" + "=" * 80)
    print("DETAILED DESIGN COMPARISON ACROSS ALL SPEEDS")
    print("=" * 80)

    # Get all speeds
    all_speeds = set()
    for design_data in data.values():
        all_speeds.update(design_data.keys())
    speeds = sorted(list(all_speeds))

    design_names = list(data.keys())

    # Print data for each speed
    for speed in speeds:
        print(f"\nSPEED: {speed} RPM")
        print("-" * 40)

        speed_data = {}
        for design_name in design_names:
            if speed in data[design_name]:
                speed_data[design_name] = data[design_name][speed]

        if not speed_data:
            print("  No data available for this speed")
            continue

        # Print individual values
        for design_name, values in speed_data.items():
            print(f"  {design_name.upper()}:")
            print(f"    Mechanical Loss: {values['mechanical_loss']:.2f} W")
            print(f"    Volumetric Loss: {values['volumetric_loss']:.2f} W")
            print(f"    Total Loss:      {values['total_loss']:.2f} W")

        # Print comparison if we have multiple designs
        if len(speed_data) >= 2:
            design_list = list(speed_data.keys())
            base_design = design_list[0]

            for i in range(1, len(design_list)):
                compare_design = design_list[i]
                print(f"\n  COMPARISON ({compare_design.upper()} vs {base_design.upper()}):")

                # Calculate differences
                mech_diff = speed_data[compare_design]['mechanical_loss'] - speed_data[base_design]['mechanical_loss']
                vol_diff = speed_data[compare_design]['volumetric_loss'] - speed_data[base_design]['volumetric_loss']
                total_diff = speed_data[compare_design]['total_loss'] - speed_data[base_design]['total_loss']

                # Calculate percentage changes
                mech_pct = (mech_diff / speed_data[base_design]['mechanical_loss']) * 100 if speed_data[base_design][
                                                                                                 'mechanical_loss'] != 0 else 0
                vol_pct = (vol_diff / speed_data[base_design]['volumetric_loss']) * 100 if speed_data[base_design][
                                                                                               'volumetric_loss'] != 0 else 0
                total_pct = (total_diff / speed_data[base_design]['total_loss']) * 100 if speed_data[base_design][
                                                                                              'total_loss'] != 0 else 0

                print(f"    Mechanical Loss: {mech_diff:+.2f} W ({mech_pct:+.1f}%)")
                print(f"    Volumetric Loss: {vol_diff:+.2f} W ({vol_pct:+.1f}%)")
                print(f"    Total Loss:      {total_diff:+.2f} W ({total_pct:+.1f}%)")

    # Print overall summary
    print(f"\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    for design_name in design_names:
        design_speeds = sorted(data[design_name].keys())
        total_losses = [data[design_name][speed]['total_loss'] for speed in design_speeds]

        print(f"\n{design_name.upper()} DESIGN:")
        print(f"  Speeds analyzed: {design_speeds}")
        print(f"  Total loss range: {min(total_losses):.2f} - {max(total_losses):.2f} W")
        print(f"  Average total loss: {np.mean(total_losses):.2f} W")


def main_flexible_design_comparison(design_paths, plot_type='absolute', pressure=320, displacement=100):
    """
    Main function to process and compare designs from custom folder paths

    Parameters:
    - design_paths: Dictionary with design names as keys and folder paths as values
                   e.g., {'industrial': 'path/to/folder1', 'optimal': 'path/to/folder2'}
    - plot_type: 'relative' for normalized percentage plots, 'absolute' for actual values
    - pressure: Fixed pressure value (default: 320 bar)
    - displacement: Fixed displacement percentage (default: 100%)
    """
    print("Processing flexible design comparison data...")
    print(f"Design paths:")
    for design_name, path in design_paths.items():
        print(f"  {design_name}: {path}")
    print(f"Parameters: Pressure={pressure} bar, Displacement={displacement}%")

    # Process the data
    data = process_design_paths_data(design_paths, pressure, displacement)

    if not data:
        print("No data found. Please check the folder paths and structure.")
        return

    print(f"\nFound data for {len(data)} designs: {list(data.keys())}")

    # Print detailed comparison
    print_detailed_comparison(data)

    # Determine if we want relative (normalized) plots
    relative = (plot_type.lower() == 'relative')

    # Create and show the comparison plot
    if relative:
        print("\nCreating Relative (Normalized) Multi-Speed Design Comparison Plot...")
    else:
        print("\nCreating Absolute Multi-Speed Design Comparison Plot...")

    try:
        fig, ax = create_multi_speed_comparison_plot(data, pressure, displacement, relative=relative)

        if fig is not None and ax is not None:
            plt.show()
            print("Multi-speed design comparison plot displayed successfully")
        else:
            print("Could not create design comparison plot")
    except Exception as e:
        print(f"Error creating design comparison plot: {e}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":
    # Define your custom design paths
    design_folder_paths = {
        'industrial': r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T05_variable speed_ speedK_1",  # Replace with actual path
        'optimal': r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run12_Optimal_value\2_para" , # Replace with actual path

        'optimal-1': r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run12_Optimal_value\3_para_dK_19.332_dZ_19.352_zeta_8",
        'optimal-2': r"Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run12_Optimal_value\5_para_dK_19.678_dZ_19.694_LKG_47_LF_32.5_zeta_4",
    }

    # Parameters for your specific case
    pressure_value = 320  # Fixed pressure in bar
    displacement_percent = 100  # Fixed displacement in %

    # For absolute value plots:
    main_flexible_design_comparison(design_folder_paths,
                                    plot_type='absolute',
                                    pressure=pressure_value,
                                    displacement=displacement_percent)

    # Uncomment below for relative percentage plots:
    # main_flexible_design_comparison(design_folder_paths,
    #                                plot_type='relative',
    #                                pressure=pressure_value,
    #                                displacement=displacement_percent)