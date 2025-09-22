import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.figsize': (10, 6),
'font.family': 'serif'
})

class CentrifugalForceAnalyzer_Research:
    """Research version - Code 1"""

    def __init__(self):
        self.mK = 0.103  # Mass piston/slipper assembly (kg)
        self.rB = 42.08  # Pitch circle radius (mm)
        self.beta_rad = np.radians(14)  # Swashplate angle (14 degrees in radians)

    def calculate_centrifugal_force(self, phi_deg, speed_rpm, gamma_deg):
        phi = math.radians(phi_deg)
        gamma = math.radians(gamma_deg)
        omega = np.pi * speed_rpm / 30  # Convert RPM to rad/s
        rB_m = self.rB / 1000.0  # Convert mm to m

        r_temp = rB_m - (2 * rB_m * math.tan(self.beta_rad) * math.tan(gamma) * (1 - math.cos(phi)))
        FwK_inclined = self.mK * (omega * omega) * r_temp

        Fwkz = FwK_inclined * math.tan(gamma)  # Z-component
        Fwky = FwK_inclined * math.cos(gamma)  # Y-component

        return {'Fwkz': Fwkz, 'Fwky': Fwky, 'speed_rpm': speed_rpm, 'gamma_deg': gamma_deg}


class CentrifugalForceAnalyzer_Literature:
    """Literature version - Code 2"""

    def __init__(self):
        self.mK = 0.103  # Mass piston/slipper assembly (kg)
        self.rB = 42.08  # Pitch circle radius (mm)
        self.rB_mean = 41.3269  # Mean radius
        self.beta_rad = np.radians(14)  # Swashplate angle (14 degrees in radians)

    def calculate_centrifugal_force(self, phi_deg, speed_rpm, gamma_deg):
        phi = math.radians(phi_deg)
        gamma = math.radians(gamma_deg)
        omega = np.pi * speed_rpm / 30  # Convert RPM to rad/s
        rB_m = self.rB_mean / 1000.0  # Convert mm to m

        FwK_inclined = self.mK * (omega * omega) * rB_m

        Fwkz = FwK_inclined * math.sin(gamma)  # Z-component
        Fwky = FwK_inclined * math.cos(gamma)  # Y-component

        return {'Fwkz': Fwkz, 'Fwky': Fwky, 'speed_rpm': speed_rpm, 'gamma_deg': gamma_deg}


def analyze_both_methods(speed_range_rpm, gamma_angles_deg, phi_angles_deg=None):
    """Analyze forces using both methods"""
    if phi_angles_deg is None:
        phi_angles_deg = [0]

    analyzer_research = CentrifugalForceAnalyzer_Research()
    analyzer_literature = CentrifugalForceAnalyzer_Literature()

    results_research = []
    results_literature = []

    for speed_rpm in speed_range_rpm:
        for gamma_deg in gamma_angles_deg:
            for phi_deg in phi_angles_deg:
                # Research method
                forces_research = analyzer_research.calculate_centrifugal_force(phi_deg, speed_rpm, gamma_deg)
                results_research.append(forces_research)

                # Literature method
                forces_literature = analyzer_literature.calculate_centrifugal_force(phi_deg, speed_rpm, gamma_deg)
                results_literature.append(forces_literature)

    return pd.DataFrame(results_research), pd.DataFrame(results_literature)


def add_arrow_annotations(ax, data_df, colors, gamma_angles, method_name, force_component='Fwkz'):
    """Add arrow annotations pointing to lines with gamma labels"""
    for i, gamma in enumerate(gamma_angles):
        if gamma == 0:  # Skip gamma = 0 as it might be at the bottom
            continue

        color = colors[i % len(colors)]
        gamma_data = data_df[data_df['gamma_deg'] == gamma]

        if not gamma_data.empty:
            # Find a good position for annotation (around 70% of the x-axis range)
            speed_range = gamma_data['speed_rpm'].max() - gamma_data['speed_rpm'].min()
            annotation_speed = gamma_data['speed_rpm'].min() + 0.7 * speed_range

            # Find the closest data point to annotation speed
            closest_idx = (gamma_data['speed_rpm'] - annotation_speed).abs().idxmin()
            x_pos = gamma_data.loc[closest_idx, 'speed_rpm']
            y_pos = gamma_data.loc[closest_idx, force_component]

            # Calculate offset for arrow
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

            # Offset the annotation to avoid overlapping with the line
            y_offset = y_range * 0.08 * (1 + i * 0.3)  # Stagger annotations vertically
            x_offset = x_range * 0.05

            # Add arrow annotation
            ax.annotate(f'γ = {gamma}°',
                        xy=(x_pos, y_pos),
                        xytext=(x_pos + x_offset, y_pos + y_offset),
                        arrowprops=dict(arrowstyle='->',
                                        connectionstyle='arc3,rad=0.1',
                                        color=color,
                                        lw=2,
                                        alpha=0.8),
                        fontsize=16,
                        fontweight='bold',
                        color=color,
                        ha='left',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.4',
                                  facecolor='white',
                                  edgecolor=color,
                                  alpha=0.9))


def plot_combined_comparison(research_df, literature_df, plot_dir='plots'):
    """Create combined plots showing both methods with matching colors and arrow annotations"""
    os.makedirs(plot_dir, exist_ok=True)

    # Define colors for each gamma angle
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Combined Plot for Fwkz
    fig1, ax1 = plt.subplots(figsize=(8, 8), facecolor='white')
    ax1.set_facecolor('white')

    gamma_angles = sorted(research_df['gamma_deg'].unique())

    # Plot research data (solid lines) and literature data (dashed lines) with same colors
    for i, gamma in enumerate(gamma_angles):
        color = colors[i % len(colors)]

        # Research data (solid lines)
        gamma_data_research = research_df[research_df['gamma_deg'] == gamma]
        ax1.plot(gamma_data_research['speed_rpm'], gamma_data_research['Fwkz'],
                 linewidth=3, linestyle='-', color=color, label=f'Research γ = {gamma}°')

        # Literature data (dashed lines) - same color
        gamma_data_literature = literature_df[literature_df['gamma_deg'] == gamma]
        ax1.plot(gamma_data_literature['speed_rpm'], gamma_data_literature['Fwkz'],
                 linewidth=3, linestyle='--', color=color, label=f'Literature γ = {gamma}°')

    # Add arrow annotations for research method (using solid lines as reference)
    add_arrow_annotations(ax1, research_df, colors, gamma_angles, 'Research', 'Fwkz')

    ax1.set_xlabel('Speed [RPM]', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Axial Component - Centrifugal Force [N]', fontsize=16, fontweight='bold')
    # ax1.set_title(
    #     'Attaching centrifugal force component: Fwkz vs Speed\n(Research: Solid Lines, Literature: Dashed Lines)',
    #     fontsize=18, fontweight='bold', pad=20)
    #
    ax1.grid(True, alpha=0.4, linewidth=1, color='gray')

    # Add custom legend entries for line styles
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=3, linestyle='-'),
                    Line2D([0], [0], color='black', lw=3, linestyle='--')]

    legend = ax1.legend(custom_lines, ['Research', 'Literature'],
                        fontsize=14, loc='upper center', ncol=2,
                        frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
    legend.get_frame().set_facecolor('white')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(f'{plot_dir}/fwkz_combined_comparison.png', dpi=300, bbox_inches='tight',
                transparent=True, facecolor='none', edgecolor='none')
    plt.show()

    # Combined Plot for Fwky
    fig2, ax2 = plt.subplots(figsize=(12, 10), facecolor='white')
    ax2.set_facecolor('white')

    # Plot research data (solid lines) and literature data (dashed lines) with same colors
    for i, gamma in enumerate(gamma_angles):
        color = colors[i % len(colors)]

        # Research data (solid lines)
        gamma_data_research = research_df[research_df['gamma_deg'] == gamma]
        ax2.plot(gamma_data_research['speed_rpm'], gamma_data_research['Fwky'],
                 linewidth=3, linestyle='-', color=color, label=f'Research γ = {gamma}°')

        # Literature data (dashed lines) - same color
        gamma_data_literature = literature_df[literature_df['gamma_deg'] == gamma]
        ax2.plot(gamma_data_literature['speed_rpm'], gamma_data_literature['Fwky'],
                 linewidth=3, linestyle='--', color=color, label=f'Literature γ = {gamma}°')

    # Add arrow annotations for research method (using solid lines as reference)
    add_arrow_annotations(ax2, research_df, colors, gamma_angles, 'Research', 'Fwky')

    ax2.set_xlabel('Speed [RPM]', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Fwky [N]', fontsize=16, fontweight='bold')
    ax2.set_title('Fwky vs Speed\n(Research: Solid Lines, Literature: Dashed Lines)',
                  fontsize=18, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.4, linewidth=1, color='gray')

    # Add custom legend entries for line styles
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=3, linestyle='-'),
                    Line2D([0], [0], color='black', lw=3, linestyle='--')]

    legend = ax2.legend(custom_lines, ['Research (Solid Lines)', 'Literature (Dashed Lines)'],
                        fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2,
                        frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
    legend.get_frame().set_facecolor('white')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(f'{plot_dir}/fwky_combined_comparison.png', dpi=300, bbox_inches='tight',
                transparent=True, facecolor='none', edgecolor='none')
    plt.show()


def create_side_by_side_comparison(research_df, literature_df, plot_dir='plots'):
    """Create side-by-side comparison plots with matching colors and arrow annotations"""
    os.makedirs(plot_dir, exist_ok=True)

    # Define colors for each gamma angle
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Create a figure with 2 subplots side by side
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8), facecolor='white')

    gamma_angles = sorted(research_df['gamma_deg'].unique())

    # Research plot (left)
    ax1.set_facecolor('white')
    for i, gamma in enumerate(gamma_angles):
        color = colors[i % len(colors)]
        gamma_data = research_df[research_df['gamma_deg'] == gamma]
        ax1.plot(gamma_data['speed_rpm'], gamma_data['Fwkz'],
                 linewidth=3, color=color, label=f'γ = {gamma}°')

    # Add arrow annotations for research plot
    add_arrow_annotations(ax1, research_df, colors, gamma_angles, 'Research', 'Fwkz')

    ax1.set_xlabel('Speed [RPM]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Axial Component - Centrifugal Force [N]', fontsize=14, fontweight='bold')
    # ax1.set_title('Research Method\n(Complex radius, tan(γ))', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4, linewidth=1, color='gray')

    # Add custom legend entries for line styles_
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=3, linestyle='-')]
    # ax1.legend(custom_lines, ['Research Method'], fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # # Literature plot (right) - using same colors
    # ax2.set_facecolor('white')
    # for i, gamma in enumerate(gamma_angles):
    #     color = colors[i % len(colors)]
    #     gamma_data = literature_df[literature_df['gamma_deg'] == gamma]
    #     ax2.plot(gamma_data['speed_rpm'], gamma_data['Fwkz'],
    #              linewidth=3, color=color, label=f'γ = {gamma}°')

    # # Add arrow annotations for literature plot
    # add_arrow_annotations(ax2, literature_df, colors, gamma_angles, 'Literature', 'Fwkz')
    #
    # ax2.set_xlabel('Speed [RPM]', fontsize=14, fontweight='bold')
    # ax2.set_ylabel('Fwkz [N]', fontsize=14, fontweight='bold')
    # ax2.set_title('Literature Method\n(Mean radius, sin(γ))', fontsize=16, fontweight='bold', pad=20)
    # ax2.grid(True, alpha=0.4, linewidth=1, color='gray')
    #
    # # Add custom legend entries for line styles
    # from matplotlib.lines import Line2D
    # custom_lines = [Line2D([0], [0], color='black', lw=3, linestyle='-')]
    # ax2.legend(custom_lines, ['Literature Method'], fontsize=14)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)

    # plt.tight_layout()
    plt.savefig(f'{plot_dir}/FwK.png', dpi=300, bbox_inches='tight',
                transparent=True, facecolor='none', edgecolor='none')
    plt.show()


def print_numerical_comparison(research_df, literature_df):
    """Print numerical comparison at key operating points"""
    print("=== NUMERICAL COMPARISON ===\n")

    # Compare at different speeds and gamma angles
    test_speeds = [1000, 3000, 5000]
    test_gammas = [5, 10, 15, 20]

    print(f"{'Speed':<8} {'Gamma':<8} {'Research':<12} {'Literature':<12} {'Diff %':<10}")
    print("-" * 60)

    for speed in test_speeds:
        for gamma in test_gammas:
            # Find closest values in dataframes
            research_row = research_df[
                (abs(research_df['speed_rpm'] - speed) < 50) &
                (research_df['gamma_deg'] == gamma)
                ]
            literature_row = literature_df[
                (abs(literature_df['speed_rpm'] - speed) < 50) &
                (literature_df['gamma_deg'] == gamma)
                ]

            if not research_row.empty and not literature_row.empty:
                research_val = research_row['Fwkz'].iloc[0]
                literature_val = literature_row['Fwkz'].iloc[0]
                diff_percent = abs(research_val - literature_val) / research_val * 100

                print(f"{speed:<8} {gamma:<8} {research_val:<12.2f} {literature_val:<12.2f} {diff_percent:<10.1f}")


# Example usage
if __name__ == "__main__":
    speed_range_rpm = np.linspace(500, 5000, 20)  # 500 to 5000 RPM, 20 points
    gamma_angles_deg = [0, 5, 10, 15, 20]  # Gamma angles in degrees

    # Analyze both methods
    research_results, literature_results = analyze_both_methods(speed_range_rpm, gamma_angles_deg)

    # Create comparison plots
    print("Creating combined comparison plots...")
    plot_combined_comparison(research_results, literature_results)

    print("\nCreating side-by-side comparison...")
    create_side_by_side_comparison(research_results, literature_results)

    # Print numerical comparison
    print_numerical_comparison(research_results, literature_results)

    print("\nPlots saved:")
    print("- fwkz_combined_comparison.png (overlaid comparison with arrows)")
    print("- fwky_combined_comparison.png (overlaid comparison with arrows)")
    print("- FwK.png (side-by-side comparison with arrows)")