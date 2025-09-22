import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# Update global styling to match previous code
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.figsize': (10, 6),
    'font.family': 'serif'
})

# File paths for length and diameter change scenarios
file_groups = {
    "Length_Change": {
        "Fluid Mesh": "Piston_ouput_data_different_design/piston_length_changed_in_input.txt",
        "Solid Mesh + Fluid Mesh": "Piston_ouput_data_different_design/piston_length_changed_in_CAD_&_input.txt",
        # "CAD_2": "Piston_ouput_data_different_design/piston_shorter_length.txt"
    },
    "Diameter_Change": {
        "Fluid Mesh": "Piston_ouput_data_different_design/piston_diameter_changed_in_input.txt",
        "Solid Mesh + Fluid Mesh": "Piston_ouput_data_different_design/piston_diameter_changed_in_CAD_&_input.txt",
        # "CAD_2": "Piston_ouput_data_different_design/piston_shorter_length.txt"
    }
}

# Line styles - Fixed to match all possible labels
line_styles = {
    "TEXT": '-',
    "CAD": '--',
    "CAD_2": '-',
    "Longer": '--',
    "Shorter": '-',
    "CASE 2": '-'
}

# Column renaming map
column_mapping = {
    'Total_Volumetric_Power_Loss': 'Total_vol_power_loss',
    'Total_Mechanical_Power_Loss': 'Total_hm_power_loss',
    'Mechanical_Power_Loss': 'single_hm_power_loss',
    'Volumetric_Power_Loss': 'single_vol_power_loss'
}

# Columns to focus on for power loss plots - Fixed column names
power_loss_columns = [
    'single_piston_vol_power_loss',
    'single_piston_hm_power_loss',
    'single_piston_total_power_loss',
    'Total_vol_power_loss',  # Fixed: was 'total_vol_power_loss'
    'Total_hm_power_loss',  # Fixed: was 'total_hm_power_loss'
    'Total_power_loss'  # Fixed: was 'total_power_loss'
]


# Helper: chunk columns
def chunk_columns(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Helper: create title based on group and comparison
def create_plot_title(group_name, column_name, comparison_labels):
    """Create descriptive title for plots"""
    if group_name == "Length_Change":
        base_title = "Length Change"
    elif group_name == "Diameter_Change":
        base_title = "Diameter Change"
    else:
        base_title = group_name.replace("_", " ")

    # Add comparison info if multiple datasets
    if len(comparison_labels) > 1:
        comparison_str = " vs ".join(comparison_labels)
        title = f"{base_title}: {comparison_str}"
    else:
        title = base_title

    return title


# Process each group (Length, Diameter)
for group_name, paths in file_groups.items():
    print(f"\n{'=' * 50}")
    print(f"Processing {group_name}")
    print(f"{'=' * 50}")

    # Create output directory for individual plots
    output_dir = f"{group_name.lower()}_individual_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Load data with error handling
    dfs = {}
    for label, path in paths.items():
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, sep='\t', engine='python')
                dfs[label] = df
                print(f"Successfully loaded {label}: {len(df)} rows, {len(df.columns)} columns")
                print(f"  Columns: {list(df.columns)}")
            else:
                print(f"⚠️ File not found: {path}")
        except Exception as e:
            print(f"⚠️ Error loading {label}: {e}")

    if not dfs:
        print(f"⚠️ No data files loaded for {group_name}. Skipping...")
        continue

    # Rename columns and add combined power loss
    for label, df in dfs.items():
        print(f"\nProcessing {label}...")

        # Apply column renaming
        df.rename(columns=column_mapping, inplace=True)
        print(f"  After renaming: {list(df.columns)}")

        # Add combined power loss columns
        if 'single_combined_power_loss' not in df.columns:
            if 'single_hm_power_loss' in df.columns and 'single_vol_power_loss' in df.columns:
                df['single_combined_power_loss'] = df['single_hm_power_loss'] + df['single_vol_power_loss']
                print("  ✓ Added single_combined_power_loss")

        if 'Total_combined_power_loss' not in df.columns:
            if 'Total_hm_power_loss' in df.columns and 'Total_vol_power_loss' in df.columns:
                df['Total_power_loss'] = df['Total_hm_power_loss'] + df['Total_vol_power_loss']
                print("  ✓ Added Total_power_loss")

    # Filter data for revolution > 5.0 and remap to 0-1 range
    for label, df in dfs.items():
        if 'revolution' not in df.columns:
            print(f"⚠️ No 'revolution' column found in {label}")
            continue

        # Filter for revolution > 5.0
        filtered_df = df[df['revolution'] > 5.0].copy()

        if len(filtered_df) > 0:
            # Remap revolution values: subtract minimum and normalize to 0-1
            min_rev = filtered_df['revolution'].min()
            max_rev = filtered_df['revolution'].max()

            if max_rev > min_rev:  # Avoid division by zero
                filtered_df['revolution'] = (filtered_df['revolution'] - min_rev) / (max_rev - min_rev)
            else:
                filtered_df['revolution'] = 0  # All values are the same

            dfs[label] = filtered_df
            print(f"  {label}: {len(dfs[label])} data points after filtering (revolution > 5.0)")
            print(f"     Original range: {min_rev:.2f} - {max_rev:.2f}, mapped to: 0.0 - 1.0")
        else:
            dfs[label] = filtered_df
            print(f"  {label}: No data points with revolution > 5.0")

    # Use available columns (excluding revolution)
    columns = [col for col in next(iter(dfs.values())).columns if col != 'revolution']
    print(f"\nAvailable columns for plotting: {columns}")

    # Filter available power loss columns
    available_power_loss_columns = [col for col in power_loss_columns if col in columns]
    print(f"Available power loss columns: {available_power_loss_columns}")

    if not available_power_loss_columns:
        print(f"⚠️ No power loss columns found for {group_name}. Skipping...")
        continue

    # Get labels for title creation
    available_labels = list(dfs.keys())

    # Power loss only comparison plots - both PDF and individual saves
    pdf_filename = f"{group_name.lower()}_power_loss_plots.pdf"
    print(f"\nCreating PDF: {pdf_filename}")

    with PdfPages(pdf_filename) as pdf:
        for chunk_idx, chunk in enumerate(chunk_columns(available_power_loss_columns, 3)):
            print(f"  Processing chunk {chunk_idx + 1}: {chunk}")

            fig, axs = plt.subplots(nrows=1, ncols=len(chunk), figsize=(18, 8))
            if len(chunk) == 1:
                axs = [axs]

            for ax, column in zip(axs, chunk):
                plot_count = 0
                for label, df in dfs.items():
                    if column in df.columns and len(df) > 0:
                        # Check if line style exists for this label
                        line_style = line_styles.get(label, '-')
                        ax.plot(df['revolution'], df[column], label=label,
                                linestyle=line_style, linewidth=2)
                        plot_count += 1
                        print(f"    Plotted {label} for {column}: {len(df)} points")

                if plot_count == 0:
                    print(f"    ⚠️ No data plotted for {column}")

                # Add title to each subplot
                plot_title = create_plot_title(group_name, column, available_labels)
                ax.set_title(f"{plot_title}\n{column}", fontweight='bold', fontsize=14, pad=20)

                # Apply consistent styling
                ax.set_xlabel("Revolution", fontweight='bold')
                ax.set_ylabel(f"{column} [W]", fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
                          frameon=True, fancybox=True, shadow=False, framealpha=0.9,
                          edgecolor='black', fontsize=12, markerscale=1.5,
                          columnspacing=2.0, handlelength=3.0, handletextpad=1.0,
                          borderpad=1.5)

            plt.tight_layout()

            # Save to PDF
            pdf.savefig(dpi=300, bbox_inches='tight')
            print(f"    ✓ Saved chunk to PDF")

            # Save individual plots for each column in this chunk
            for column in chunk:
                print(f"    Creating individual plot for {column}")

                fig_individual = plt.figure(figsize=(10, 8))
                ax_individual = fig_individual.add_subplot(111)

                plot_count = 0
                for label, df in dfs.items():
                    if column in df.columns and len(df) > 0:
                        line_style = line_styles.get(label, '-')
                        ax_individual.plot(df['revolution'], df[column], label=label,
                                           linestyle=line_style, linewidth=2)
                        plot_count += 1

                if plot_count > 0:
                    # Add title to individual plot
                    plot_title = create_plot_title(group_name, column, available_labels)
                    op_con= "4500rpm, 320bar, 100%"
                    ax_individual.set_title(f"{plot_title}\n{op_con}", fontweight='bold', fontsize=16, pad=20)

                    # Apply consistent styling
                    ax_individual.set_xlabel("Revolution", fontweight='bold')
                    ax_individual.set_ylabel(f"{column} [W]", fontweight='bold')
                    ax_individual.grid(True, alpha=0.3)
                    ax_individual.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
                                         frameon=True, fancybox=True, shadow=False, framealpha=0.9,
                                         edgecolor='black', fontsize=12, markerscale=1.5,
                                         columnspacing=2.0, handlelength=3.0, handletextpad=1.0,
                                         borderpad=1.5)

                    # Save individual plot
                    individual_filename = f"{output_dir}/{group_name.lower()}_{column}_vs_revolution.png"
                    plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
                    print(f"      ✓ Saved: {individual_filename}")

                    individual_jpg_filename = f"{group_name.lower()}_{column}_vs_revolution.jpg"
                    plt.savefig(individual_jpg_filename, dpi=300, bbox_inches='tight')
                    print(f"      ✓ Saved: {individual_jpg_filename}")
                else:
                    print(f"      ⚠️ No data to plot for {column}")

                plt.close(fig_individual)

            plt.close(fig)

    print(f"\n✓ Generated files for {group_name}:")
    print(f"   - {pdf_filename}")
    print(f"   - Individual plots saved in: {output_dir}/")
    print(f"   - Power loss plots also saved as individual JPG files")

print(f"\n{'=' * 50}")
print("Processing completed!")
print(f"{'=' * 50}")