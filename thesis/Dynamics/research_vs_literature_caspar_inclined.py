import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_friction_data(research_folder_path, literature_folder_path, gamma_values=[0, 5, 10, 15]):
    """
    Load piston force data from research (with friction) and literature (no friction) sources.
    """
    research_data_dict = {}
    literature_data_dict = {}

    print("Loading research data...")
    for gamma in gamma_values:
        file_path = os.path.join(research_folder_path, f"piston_{gamma}.txt")
        try:
            data = pd.read_csv(file_path, sep='\t')
            research_data_dict[gamma] = data
            print(f"Loaded research piston_{gamma}.txt shape: {data.shape}")
        except Exception as e:
            print(f"Warning: Could not load research piston_{gamma}.txt: {e}")
            research_data_dict[gamma] = pd.DataFrame()

    print("\nLoading literature data...")
    for gamma in gamma_values:
        file_path = os.path.join(literature_folder_path, f"piston_{gamma}.txt")
        try:
            data = pd.read_csv(file_path, sep='\t')
            literature_data_dict[gamma] = data
            print(f"Loaded literature piston_{gamma}.txt shape: {data.shape}")
        except Exception as e:
            print(f"Warning: Could not load literature piston_{gamma}.txt: {e}")
            literature_data_dict[gamma] = pd.DataFrame()

    return research_data_dict, literature_data_dict


def filter_piston_data(data_dict, label, max_revolution=1.0):
    """
    Filter piston data to revolution <= max and convert revolution to degrees.
    """
    filtered_dict = {}
    for gamma, df in data_dict.items():
        if not df.empty:
            df_filtered = df[df['revolution'] <= max_revolution].copy()
            df_filtered['phi_deg_piston'] = df_filtered['revolution'] * 360.0
            filtered_dict[gamma] = df_filtered
            print(f"{label} piston_{gamma} filtered: {len(df_filtered)} rows")
        else:
            filtered_dict[gamma] = pd.DataFrame()
            print(f"{label} piston_{gamma} has no data.")
    return filtered_dict


def add_arrow_annotations_inertial(ax, research_data, literature_data, gamma_colors, gamma_values):
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'legend.fontsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.figsize': (10, 6),
        'font.family': 'serif'
    })

    """Add arrow annotations pointing to lines with gamma labels for inertial force"""
    for gamma in gamma_values:
        if gamma == 0:  # Skip gamma = 0 to avoid cluttering
            continue

        color = gamma_colors.get(gamma, 'gray')

        # Use research data as reference for arrow positioning
        df_research = research_data.get(gamma, pd.DataFrame())

        if not df_research.empty and 'FaK' in df_research.columns:
            # Find a good position for annotation (around 60% of the x-axis range)
            phi_range = df_research['phi_deg_piston'].max() - df_research['phi_deg_piston'].min()
            annotation_phi = df_research['phi_deg_piston'].min() + 0.6 * phi_range

            # Find the closest data point to annotation position
            closest_idx = (df_research['phi_deg_piston'] - annotation_phi).abs().idxmin()
            x_pos = df_research.loc[closest_idx, 'phi_deg_piston']
            y_pos = df_research.loc[closest_idx, 'FaK']

            # Calculate offset for arrow
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

            # Offset the annotation to avoid overlapping with the line
            y_offset = y_range * 0.15 * (1 + gamma * 0.2)  # Stagger annotations vertically
            x_offset = x_range * 0.08

            # Add arrow annotation
            ax.annotate(f'γ = {gamma}°',
                        xy=(x_pos, y_pos),
                        xytext=(x_pos + x_offset, y_pos + y_offset),
                        arrowprops=dict(arrowstyle='->',
                                        connectionstyle='arc3,rad=0.2',
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


def create_individual_plots_enhanced(research_data, literature_data, force_columns, output_dir):
    """
    Create and save individual comparison plots with enhanced styling for inertial force.
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'legend.fontsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.figsize': (10, 6),
        'font.family': 'serif'
    })
    os.makedirs(output_dir, exist_ok=True)

    gamma_colors = {0: 'red', 5: 'green', 10: 'orange', 15: 'purple'}

    # Define descriptive names for force columns
    force_descriptions = {
        'FaK': 'Inertial Force',
        'FAK_inclined': 'Total Piston Force',
        'FAKy': 'Radial Component of Total Piston Force',
        'Fwkz': 'Attaching Centrifugal Force',
        'FSKy': 'Radial Component of Reaction Force'
    }

    for force_col in force_columns:
        # Create figure with professional styling
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white', dpi=300)
        ax.set_facecolor('white')

        for gamma in sorted(research_data.keys()):
            color = gamma_colors.get(gamma, 'gray')

            df_research = research_data.get(gamma, pd.DataFrame())
            df_literature = literature_data.get(gamma, pd.DataFrame())

            if not df_research.empty and force_col in df_research.columns:
                ax.plot(df_research['phi_deg_piston'], df_research[force_col],
                        color=color, linestyle='-', linewidth=3)

            if not df_literature.empty and force_col in df_literature.columns:
                ax.plot(df_literature['phi_deg_piston'], df_literature[force_col],
                        color=color, linestyle='--', linewidth=3)

        # Special treatment for inertial force (FaK) - add arrows and simplified legend
        if force_col == 'FaK':
            # Add arrow annotations
            gamma_values = sorted([g for g in research_data.keys() if not research_data[g].empty])
            add_arrow_annotations_inertial(ax, research_data, literature_data, gamma_colors, gamma_values)

            # Create simplified legend with larger font
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='black', lw=3, linestyle='-'),
                            Line2D([0], [0], color='black', lw=3, linestyle='--')]

            legend = ax.legend(custom_lines, ['Research-study', 'Literature'],
                               fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2,
                               frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
            legend.get_frame().set_facecolor('white')
            ax.set_ylabel('Inertial Force[N]', fontsize=16, fontweight='bold')
        else:
            # Original legend for other forces
            legend = ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3,
                               frameon=True, fancybox=True, shadow=False, framealpha=0.8, edgecolor='black')
            legend.get_frame().set_facecolor('white')
            ax.set_ylabel(f'{force_col} [N]', fontsize=16, fontweight='bold')

        # Use descriptive title with force description - professional formatting
        force_description = force_descriptions.get(force_col, force_col)
        # ax.set_title(f'{force_description} ({force_col})\nResearch vs Literature',
        #              fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Shaft angle [°]', fontsize=16, fontweight='bold')
        # ax.set_ylabel(f'{force_col} [N]', fontsize=16, fontweight='bold')

        # Set ticks and limits
        ax.set_xticks(range(0, 361, 60))
        ax.set_xlim(0, 360)

        # Grid styling
        ax.grid(True, alpha=0.4, linewidth=1, color='gray')

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        plt.tight_layout()

        # Save plot with transparent background for PowerPoint flexibility
        output_file = os.path.join(output_dir, f'{force_col}_comparison.png')
        output_file = os.path.join(output_dir, f'{force_col}_comparison.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight',
                    transparent=True, facecolor='none', edgecolor='none')
        plt.close(fig)
        print(f"Saved plot: {output_file}")


# Usage: Replace your create_individual_plots function with create_individual_plots_enhanced
# In your main() function, change this line:
# create_individual_plots(research_filtered, literature_filtered, force_columns, output_dir)
# To:
# create_individual_plots_enhanced(research_filtered, literature_filtered, force_columns, output_dir)
def print_numerical_comparison(research_data, literature_data, force_columns, gamma_values):
    """
    Print comprehensive numerical comparison at key operating points for all forces and gamma values.

    Parameters:
    - research_data: Dictionary with gamma as keys and DataFrames as values
    - literature_data: Dictionary with gamma as keys and DataFrames as values
    - force_columns: List of force column names to compare
    - gamma_values: List of gamma values to analyze
    """
    print("=" * 80)
    print("COMPREHENSIVE NUMERICAL COMPARISON - RESEARCH vs LITERATURE")
    print("=" * 80)

    # Define key revolution points for comparison (in degrees)
    test_revolutions = [0, 60, 120, 180, 240, 300]

    # Force descriptions for better readability
    force_descriptions = {
        'FaK': 'Inertial Force',
        'FAK_inclined': 'Total Piston Force',
        'FAKy': 'Radial Component of Total Piston Force',
        'Fwkz': 'Attaching Centrifugal Force',
        'FSKy': 'Radial Component of Reaction Force'
    }

    for force_col in force_columns:
        force_description = force_descriptions.get(force_col, force_col)
        print(f"\n{'=' * 20} {force_description} ({force_col}) {'=' * 20}")

        for gamma in gamma_values:
            research_df = research_data.get(gamma, pd.DataFrame())
            literature_df = literature_data.get(gamma, pd.DataFrame())

            if research_df.empty or literature_df.empty:
                print(f"\nGamma = {gamma}°: No data available")
                continue

            if force_col not in research_df.columns or force_col not in literature_df.columns:
                print(f"\nGamma = {gamma}°: Force column '{force_col}' not found")
                continue

            print(f"\nGamma = {gamma}°:")
            print(f"{'Revolution':<12} {'Research':<15} {'Literature':<15} {'Abs Diff':<12} {'Rel Diff %':<12}")
            print("-" * 75)

            total_abs_diff = 0
            total_rel_diff = 0
            valid_comparisons = 0

            for revolution in test_revolutions:
                # Find closest revolution values in dataframes (within ±5 degrees tolerance)
                research_rows = research_df[
                    abs(research_df['phi_deg_piston'] - revolution) <= 5
                    ]
                literature_rows = literature_df[
                    abs(literature_df['phi_deg_piston'] - revolution) <= 5
                    ]

                if not research_rows.empty and not literature_rows.empty:
                    # Take the closest match
                    research_idx = \
                    research_rows.iloc[(research_rows['phi_deg_piston'] - revolution).abs().argsort()[:1]].index[0]
                    literature_idx = \
                    literature_rows.iloc[(literature_rows['phi_deg_piston'] - revolution).abs().argsort()[:1]].index[0]

                    research_val = research_df.loc[research_idx, force_col]
                    literature_val = literature_df.loc[literature_idx, force_col]

                    abs_diff = abs(research_val - literature_val)

                    # Calculate relative difference (avoid division by zero)
                    if abs(research_val) > 1e-6:  # Avoid very small denominators
                        rel_diff_percent = abs_diff / abs(research_val) * 100
                    else:
                        rel_diff_percent = float('inf') if abs_diff > 1e-6 else 0

                    print(
                        f"{revolution:<12}° {research_val:<15.2f} {literature_val:<15.2f} {abs_diff:<12.2f} {rel_diff_percent:<12.1f}")

                    # Accumulate for statistics
                    total_abs_diff += abs_diff
                    if rel_diff_percent != float('inf'):
                        total_rel_diff += rel_diff_percent
                        valid_comparisons += 1
                else:
                    print(f"{revolution:<12}° {'N/A':<15} {'N/A':<15} {'N/A':<12} {'N/A':<12}")

            # Print summary statistics for this gamma
            if valid_comparisons > 0:
                avg_abs_diff = total_abs_diff / len(test_revolutions)
                avg_rel_diff = total_rel_diff / valid_comparisons
                print(f"{'Average:':<12} {'':<15} {'':<15} {avg_abs_diff:<12.2f} {avg_rel_diff:<12.1f}")

            # Find maximum differences
            if not research_df.empty and not literature_df.empty and force_col in research_df.columns and force_col in literature_df.columns:
                max_research = research_df[force_col].max()
                min_research = research_df[force_col].min()
                max_literature = literature_df[force_col].max()
                min_literature = literature_df[force_col].min()

                print(
                    f"{'Max Value:':<12} {max_research:<15.2f} {max_literature:<15.2f} {abs(max_research - max_literature):<12.2f}")
                print(
                    f"{'Min Value:':<12} {min_research:<15.2f} {min_literature:<15.2f} {abs(min_research - min_literature):<12.2f}")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall summary table
    print(f"\n{'Force':<20} {'Gamma':<8} {'Max Abs Diff':<15} {'Max Rel Diff %':<15}")
    print("-" * 65)

    for force_col in force_columns:
        force_description = force_descriptions.get(force_col, force_col)

        for gamma in gamma_values:
            research_df = research_data.get(gamma, pd.DataFrame())
            literature_df = literature_data.get(gamma, pd.DataFrame())

            if (not research_df.empty and not literature_df.empty and
                    force_col in research_df.columns and force_col in literature_df.columns):

                # Calculate maximum differences across all data points
                research_vals = research_df[force_col].values
                literature_vals = literature_df[force_col].values

                # Interpolate to same length for comparison if needed
                if len(research_vals) != len(literature_vals):
                    min_len = min(len(research_vals), len(literature_vals))
                    research_vals = research_vals[:min_len]
                    literature_vals = literature_vals[:min_len]

                abs_diffs = np.abs(research_vals - literature_vals)
                max_abs_diff = np.max(abs_diffs)

                # Calculate relative differences (avoid division by zero)
                rel_diffs = []
                for r_val, l_val in zip(research_vals, literature_vals):
                    if abs(r_val) > 1e-6:
                        rel_diffs.append(abs(r_val - l_val) / abs(r_val) * 100)

                max_rel_diff = np.max(rel_diffs) if rel_diffs else 0

                print(f"{force_col:<20} {gamma:<8}° {max_abs_diff:<15.2f} {max_rel_diff:<15.1f}")
            else:
                print(f"{force_col:<20} {gamma:<8}° {'N/A':<15} {'N/A':<15}")


def main():
    """
    Main function to compare research and literature piston force data with numerical comparison.
    """
    # Input paths
    research_folder = "C:/Users/MIT/Desktop/thesis_temp/Results_inclined_piston_forces_caspar/Results/Friction"
    literature_folder = "C:/Users/MIT/Desktop/thesis_temp/Results_inclined_piston_forces_caspar/Results/literature"

    # Output path
    output_dir = "C:/Users/MIT/Desktop/thesis_temp/comparison_literature_vs_research"

    # Columns to compare
    force_columns = ['FaK', 'FAK_inclined', 'FAKy', 'Fwkz', 'FSKy']
    gamma_values = [0, 5, 10, 15]

    # Load data
    research_data, literature_data = load_friction_data(research_folder, literature_folder, gamma_values)

    # Filter
    research_filtered = filter_piston_data(research_data, "Research", max_revolution=1.0)
    literature_filtered = filter_piston_data(literature_data, "Literature", max_revolution=1.0)

    # Create plots
    print("Generating comparison plots...")
    # create_individual_plots(research_filtered, literature_filtered, force_columns, output_dir)
    create_individual_plots_enhanced(research_filtered, literature_filtered, force_columns, output_dir)
    # Print numerical comparison
    print_numerical_comparison(research_filtered, literature_filtered, force_columns, gamma_values)


if __name__ == "__main__":
    main()