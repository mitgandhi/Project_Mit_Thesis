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
                          label=f'Vol Loss - Zeta {zeta}°',
                          color=light_color,
                          edgecolor='white', linewidth=0.5)

        # Top bars (mechanical loss) - dark color, stacked on volumetric
        mech_bars = ax.bar(pos, mech_values, width, bottom=vol_values,
                           label=f'Mech Loss - Zeta {zeta}°',
                           color=dark_color,
                           edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for j, (vol_val, mech_val) in enumerate(zip(vol_values, mech_values)):
            total_val = vol_val + mech_val
            if total_val > 0:
                # Label for total value at the top
                if relative:
                    ax.text(pos[j], total_val + (max(vol_values + mech_values) * 0.01),
                            f'{total_val:.1f}%', ha='center', va='bottom',
                            fontweight='bold', fontsize=9)
                else:
                    ax.text(pos[j], total_val + (max(vol_values + mech_values) * 0.01),
                            f'{total_val:.0f}W', ha='center', va='bottom',
                            fontweight='bold', fontsize=9)

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
        f'{title_prefix}Power Loss Breakdown\nPressure: {pressure_val} bar, Displacement: {displacement_val} %')

    # Create custom legend to show volumetric (light) and mechanical (dark) losses
    legend_elements = []
    for i, zeta in enumerate(zeta_values):
        base_color = base_colors[i % len(base_colors)]

        # Light patch for volumetric
        from matplotlib.patches import Patch
        light_color = list(to_rgba(base_color))
        light_color[3] = 0.4
        legend_elements.append(Patch(facecolor=light_color, edgecolor='white',
                                     label=f'Vol - Zeta {zeta}°'))

        # Dark patch for mechanical
        dark_color = list(to_rgba(base_color))
        dark_color[3] = 0.9
        legend_elements.append(Patch(facecolor=dark_color, edgecolor='white',
                                     label=f'Mech - Zeta {zeta}°'))

    # Legend below x-axis
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25),
              ncol=4, frameon=True, fancybox=True, shadow=False, framealpha=0.9,
              edgecolor='black', fontsize=10, markerscale=1.2, columnspacing=1.5,
              handlelength=2.0, handletextpad=0.5, borderpad=1.0)

    # Grid
    ax.grid(True, alpha=0.3, axis='y')

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the figure
    fig.savefig('power_loss_analysis_stacked.png', dpi=300, bbox_inches='tight')

    # Return the figure and axis objects
    return fig, ax


def create_efficiency_bar_chart(efficiency_data, efficiency_type='overall', pressure_val=None, displacement_val=None):
    """
    Create a bar chart for efficiency analysis
    X-axis: Zeta values (Piston inclination angle), Colored bars: Operating Conditions (speeds)
    Y-axis: Efficiency in %

    efficiency_type: 'overall', 'volumetric', 'hydromech', or 'all'
    """
    # Extract unique speed values and sort them
    all_speeds = set()
    for zeta_data in efficiency_data.values():
        all_speeds.update(zeta_data.keys())
    speeds = sorted(list(all_speeds))

    # Extract zeta values and sort them
    zeta_values = sorted(efficiency_data.keys())

    if efficiency_type == 'all':
        # Create subplot for all efficiency types
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        axes = [ax1, ax2, ax3, ax4]
        eff_types = ['overall_eff', 'volumetric_eff', 'hydromech_eff']
        titles = ['Overall Efficiency', 'Volumetric Efficiency', 'Hydro-Mechanical Efficiency', 'Efficiency Comparison']

        # Define colors for different operating conditions (speeds)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        x = np.arange(len(zeta_values))
        width = 0.15  # Width of bars for grouped bar chart

        # Plot first three efficiency types as grouped bar charts
        for plot_idx, (ax, eff_type, title) in enumerate(zip(axes[:3], eff_types, titles[:3])):
            for i, speed in enumerate(speeds):
                color = colors[i % len(colors)]
                values = []
                for zeta in zeta_values:
                    if speed in efficiency_data[zeta]:
                        values.append(efficiency_data[zeta][speed][eff_type])
                    else:
                        values.append(0)

                # Offset bars for each speed
                x_offset = x + (i - len(speeds) / 2 + 0.5) * width
                bars = ax.bar(x_offset, values, width, label=f'OC{i + 1} ({speed} RPM)',
                              color=color, alpha=0.8)

                # Add value labels on bars
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, rotation=0)

            # Formatting
            ax.set_xlabel('Piston Inclination Angle (°)')
            ax.set_ylabel('Efficiency (%)')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{zeta}' for zeta in zeta_values])
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Set appropriate y-limits based on efficiency type
            if eff_type == 'overall_eff':
                ax.set_ylim(70, 100)
            elif eff_type == 'volumetric_eff':
                ax.set_ylim(85, 105)
            else:  # hydromech_eff
                ax.set_ylim(70, 105)

        # Fourth plot: Average efficiency comparison across all speeds
        ax4 = axes[3]
        avg_overall = []
        avg_volumetric = []
        avg_hydromech = []

        for zeta in zeta_values:
            overall_vals = [efficiency_data[zeta][speed]['overall_eff']
                            for speed in speeds if speed in efficiency_data[zeta]]
            vol_vals = [efficiency_data[zeta][speed]['volumetric_eff']
                        for speed in speeds if speed in efficiency_data[zeta]]
            hm_vals = [efficiency_data[zeta][speed]['hydromech_eff']
                       for speed in speeds if speed in efficiency_data[zeta]]

            avg_overall.append(np.mean(overall_vals) if overall_vals else 0)
            avg_volumetric.append(np.mean(vol_vals) if vol_vals else 0)
            avg_hydromech.append(np.mean(hm_vals) if hm_vals else 0)

        width_comp = 0.25
        x_overall = x - width_comp
        x_volumetric = x
        x_hydromech = x + width_comp

        bars1 = ax4.bar(x_overall, avg_overall, width_comp, label='Overall Efficiency',
                        color='green', alpha=0.8)
        bars2 = ax4.bar(x_volumetric, avg_volumetric, width_comp, label='Volumetric Efficiency',
                        color='red', alpha=0.8)
        bars3 = ax4.bar(x_hydromech, avg_hydromech, width_comp, label='Hydro-Mechanical Efficiency',
                        color='blue', alpha=0.8)

        # Add value labels
        for bars, values in [(bars1, avg_overall), (bars2, avg_volumetric), (bars3, avg_hydromech)]:
            for bar, val in zip(bars, values):
                if val > 0:
                    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

        ax4.set_xlabel('Piston Inclination Angle (°)')
        ax4.set_ylabel('Average Efficiency (%)')
        ax4.set_title('Average Efficiency Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{zeta}' for zeta in zeta_values])
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best')
        ax4.set_ylim(70, 105)

        # Overall title
        if pressure_val and displacement_val:
            fig.suptitle(f'Efficiency Analysis\nPressure: {pressure_val} bar, Displacement: {displacement_val} cc/rev',
                         fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Efficiency Analysis', fontsize=16, fontweight='bold')

        plt.tight_layout()
        return fig, axes

    else:
        # Single efficiency type plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define colors for different operating conditions (speeds)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        x = np.arange(len(zeta_values))
        width = 0.15  # Width of bars

        # Map efficiency type to data key
        eff_key_map = {
            'overall': 'overall_eff',
            'volumetric': 'volumetric_eff',
            'hydromech': 'hydromech_eff'
        }

        eff_key = eff_key_map.get(efficiency_type, 'overall_eff')

        for i, speed in enumerate(speeds):
            color = colors[i % len(colors)]
            values = []
            for zeta in zeta_values:
                if speed in efficiency_data[zeta]:
                    values.append(efficiency_data[zeta][speed][eff_key])
                else:
                    values.append(0)

            # Offset bars for each speed
            x_offset = x + (i - len(speeds) / 2 + 0.5) * width
            bars = ax.bar(x_offset, values, width, label=f'OC{i + 1} ({speed} RPM)',
                          color=color, alpha=0.8)

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, rotation=0)

        # Formatting
        ax.set_xlabel('Piston Inclination Angle (°)')
        ax.set_ylabel('Efficiency (%)')

        # Set title based on efficiency type
        title_map = {
            'overall': 'Overall Efficiency',
            'volumetric': 'Volumetric Efficiency',
            'hydromech': 'Hydro-Mechanical Efficiency'
        }
        title = title_map.get(efficiency_type, 'Efficiency Analysis')

        if pressure_val and displacement_val:
            ax.set_title(f'{title}\nPressure: {pressure_val} bar, Displacement: {displacement_val} cc/rev')
        else:
            ax.set_title(title)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{zeta}' for zeta in zeta_values])
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

        # Set appropriate y-limits based on efficiency type
        if efficiency_type == 'overall':
            ax.set_ylim(70, 100)
        elif efficiency_type == 'volumetric':
            ax.set_ylim(85, 105)
        else:  # hydromech
            ax.set_ylim(70, 105)

        plt.tight_layout()
        return fig, ax


def create_efficiency_stacked_bar_chart(efficiency_data, pressure_val=None, displacement_val=None):
    """
    Create a stacked bar chart showing contribution of volumetric and hydro-mechanical efficiency
    to overall efficiency for each operating condition
    """
    # Extract unique speed values and sort them
    all_speeds = set()
    for zeta_data in efficiency_data.values():
        all_speeds.update(zeta_data.keys())
    speeds = sorted(list(all_speeds))

    # Extract zeta values and sort them
    zeta_values = sorted(efficiency_data.keys())

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for different operating conditions (speeds)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    x = np.arange(len(zeta_values))
    width = 0.6

    # For stacked bars, we'll show the breakdown of efficiency losses
    # (100% - volumetric_eff) and (100% - hydromech_eff) as loss components

    # Prepare data for stacked representation
    vol_losses = []  # Volumetric losses (100 - vol_eff)
    hm_losses = []  # Hydro-mechanical losses (100 - hm_eff)

    for zeta in zeta_values:
        vol_loss_for_zeta = []
        hm_loss_for_zeta = []
        for speed in speeds:
            if speed in efficiency_data[zeta]:
                vol_eff = efficiency_data[zeta][speed]['volumetric_eff']
                hm_eff = efficiency_data[zeta][speed]['hydromech_eff']
                vol_loss_for_zeta.append(100 - vol_eff)
                hm_loss_for_zeta.append(100 - hm_eff)
            else:
                vol_loss_for_zeta.append(0)
                hm_loss_for_zeta.append(0)
        vol_losses.append(vol_loss_for_zeta)
        hm_losses.append(hm_loss_for_zeta)

    # Convert to numpy arrays
    vol_losses = np.array(vol_losses)
    hm_losses = np.array(hm_losses)

    # Create stacked bars showing efficiency losses
    bottom = np.zeros(len(zeta_values))

    for i, speed in enumerate(speeds):
        color = colors[i % len(colors)]

        # Stack volumetric losses
        vol_values = vol_losses[:, i]
        ax.bar(x, vol_values, width, bottom=bottom,
               label=f'Vol Loss OC{i + 1} ({speed} RPM)', color=color, alpha=0.6)

        # Stack hydro-mechanical losses on top
        hm_values = hm_losses[:, i]
        ax.bar(x, hm_values, width, bottom=bottom + vol_values,
               label=f'HM Loss OC{i + 1} ({speed} RPM)', color=color, alpha=0.9)

        # Add efficiency labels
        for j, (vol_val, hm_val) in enumerate(zip(vol_values, hm_values)):
            total_loss = vol_val + hm_val
            overall_eff = 100 - total_loss
            if overall_eff > 0:
                ax.text(j, bottom[j] + total_loss + 1, f'{overall_eff:.1f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

        bottom += vol_values + hm_values

    # Formatting
    ax.set_xlabel('Piston Inclination Angle (°)')
    ax.set_ylabel('Efficiency Loss (%)')
    ax.set_ylim(0, 35)  # Assuming max loss of 35%

    ax.set_xticks(x)
    ax.set_xticklabels([f'{zeta}' for zeta in zeta_values])

    # Title
    if pressure_val and displacement_val:
        ax.set_title(
            f'Efficiency Loss Breakdown\nPressure: {pressure_val} bar, Displacement: {displacement_val} cc/rev')
    else:
        ax.set_title('Efficiency Loss Breakdown')

    # Legend
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def main(root_path, plot_type='relative', displacement_volume=56):
    """
    Main function to process data and create visualization
    plot_type: 'relative' for normalized percentage plots, 'absolute' for actual values
    displacement_volume: pump displacement in cc/rev (default 56cc for V60N)
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

    # Calculate efficiencies
    print("\nCalculating efficiencies...")
    efficiency_data = calculate_efficiencies(data, displacement_volume)

    # Determine if we want relative (normalized) plots
    relative = (plot_type.lower() == 'relative')

    # Create and show the total power loss plot
    if relative:
        print("\nCreating Relative (Normalized) Total Power Loss Plot...")
    else:
        print("\nCreating Absolute Total Power Loss Plot...")

    try:
        fig1, ax1 = create_total_power_loss_plot(data, pressure_val, displacement_val, relative=relative)
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

    # Uncomment below sections if you want to create efficiency plots as well

    # Create and show efficiency bar charts
    # print("\nCreating Efficiency Bar Charts...")
    #
    # # Option 1: All efficiency types in one figure (4 subplots)
    # try:
    #     fig2, axes = create_efficiency_bar_chart(efficiency_data, 'all', pressure_val, displacement_val)
    #     if fig2 is not None and axes is not None:
    #         plt.figure(fig2.number)
    #         plt.show()
    #         print("All efficiency plots displayed successfully")
    # except Exception as e:
    #     print(f"?? Error creating all efficiency plots: {e}")

    # Option 2: Individual efficiency bar charts
    # efficiency_types = ['overall', 'volumetric', 'hydromech']
    # for eff_type in efficiency_types:
    #     try:
    #         print(f"\nCreating {eff_type.title()} Efficiency Bar Chart...")
    #         fig, ax = create_efficiency_bar_chart(efficiency_data, eff_type, pressure_val, displacement_val)
    #         if fig is not None and ax is not None:
    #             plt.figure(fig.number)
    #             plt.show()
    #             print(f"{eff_type.title()} efficiency plot displayed successfully")
    #     except Exception as e:
    #         print(f"?? Error creating {eff_type} efficiency plot: {e}")
    #
    # Option 3: Efficiency loss breakdown (stacked bar chart)
    # try:
    #     print("\nCreating Efficiency Loss Breakdown Chart...")
    #     fig3, ax3 = create_efficiency_stacked_bar_chart(efficiency_data, pressure_val, displacement_val)
    #     if fig3 is not None and ax3 is not None:
    #         plt.figure(fig3.number)
    #         plt.show()
    #         print("Efficiency loss breakdown plot displayed successfully")
    # except Exception as e:
    #     print(f"?? Error creating efficiency loss breakdown plot: {e}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\n=== POWER LOSSES ===")
    for zeta in sorted(data.keys()):
        print(f"\nZeta {zeta}:")
        for speed in sorted(data[zeta].keys()):
            mech = data[zeta][speed]['mechanical_loss']
            vol = data[zeta][speed]['volumetric_loss']
            total = data[zeta][speed]['total_loss']
            print(f"  Speed {speed}: Mech={mech:.2f}W, Vol={vol:.2f}W, Total={total:.2f}W")

    print("\n=== EFFICIENCIES ===")
    for zeta in sorted(efficiency_data.keys()):
        print(f"\nZeta {zeta}:")
        for speed in sorted(efficiency_data[zeta].keys()):
            eff_data = efficiency_data[zeta][speed]
            print(f"  Speed {speed}: Overall={eff_data['overall_eff']:.1f}%, " +
                  f"Vol={eff_data['volumetric_eff']:.1f}%, " +
                  f"HM={eff_data['hydromech_eff']:.1f}%")

    # Print efficiency summary
    print("\n=== EFFICIENCY SUMMARY ===")
    for zeta in sorted(efficiency_data.keys()):
        overall_effs = [efficiency_data[zeta][speed]['overall_eff']
                        for speed in efficiency_data[zeta].keys()]
        vol_effs = [efficiency_data[zeta][speed]['volumetric_eff']
                    for speed in efficiency_data[zeta].keys()]
        hm_effs = [efficiency_data[zeta][speed]['hydromech_eff']
                   for speed in efficiency_data[zeta].keys()]

        if overall_effs:
            print(f"Zeta {zeta}: Avg Overall={np.mean(overall_effs):.1f}%, " +
                  f"Avg Vol={np.mean(vol_effs):.1f}%, " +
                  f"Avg HM={np.mean(hm_effs):.1f}%")


if __name__ == "__main__":
    # Replace this with the actual path to your data folder
    root_folder_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run3_V60N_with_different_inclination"

    # For relative (normalized percentage) plots:
    # main(root_folder_path, plot_type='relative')

    # For absolute value plots:
    main(root_folder_path, plot_type='absolute', displacement_volume=56)
    print(
        "Please update the root_folder_path variable with your actual data folder path and uncomment the main() call.")


