import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===== PLOT STYLE CONFIGURATION =====
PLOT_STYLE = {
    # Font sizes
    'axis_labels': 20,  # X and Y axis labels (Phi [°], Force [N])
    'axis_numbers': 24,  # Numbers on the axes (0, 90, 180, 270, etc.)
    'title': 20,  # Plot titles
    'legend': 24,  # Legend text

    # Figure properties
    'figure_size': (10, 10),  # Width, height in inches
    'line_width': 2.5,  # Line thickness
    'dpi': 300,  # Image resolution

    # Grid and transparency
    'grid_alpha': 0.3,  # Grid transparency
    'legend_alpha': 0.8,  # Legend background transparency
}


def apply_plot_style(ax, xlabel, ylabel, title):
    """Apply consistent styling to a plot"""
    ax.set_xlabel(xlabel, fontsize=PLOT_STYLE['axis_labels'], fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=PLOT_STYLE['axis_labels'], fontweight='bold')
    ax.set_title(title, fontsize=PLOT_STYLE['title'], fontweight='bold')
    ax.grid(True, alpha=PLOT_STYLE['grid_alpha'])
    ax.set_xlim(0, 360)

    # Apply axis number styling
    ax.tick_params(axis='both', which='major',
                   labelsize=PLOT_STYLE['axis_numbers'],
                   labelbottom=True, labeltop=False)
    ax.tick_params(axis='x', bottom=True, top=False)
    ax.xaxis.set_label_position('bottom')


def create_legend(ax):
    """Create consistent legend styling"""
    legend = ax.legend(fontsize=PLOT_STYLE['legend'],
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.15),
                       ncol=2,
                       frameon=True,
                       fancybox=True,
                       shadow=False,
                       framealpha=PLOT_STYLE['legend_alpha'],
                       edgecolor='black')
    legend.get_frame().set_facecolor('white')
    return legend


def save_plot(fig, output_dir, filename):
    """Save plot with consistent settings"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_STYLE['dpi'], bbox_inches='tight', transparent=True)
    print(f"Saved: {plot_path}")
    plt.close()


def load_data(piston_folder_path, csv_file_path, zeta_values=[0, 5]):
    """Load piston and CSV data"""
    piston_data_dict = {}

    # Load piston files
    for zeta in zeta_values:
        piston_file = f"piston_{zeta}.txt"
        piston_file_path = os.path.join(piston_folder_path, piston_file)

        try:
            piston_data = pd.read_csv(piston_file_path, sep='\t')
            # Filter to one revolution and convert to degrees
            piston_data = piston_data[piston_data['revolution'] <= 1.0].copy()
            piston_data['phi_deg_piston'] = piston_data['revolution'] * 360.0
            piston_data_dict[zeta] = piston_data
        except:
            piston_data_dict[zeta] = pd.DataFrame()

    # Load CSV data
    csv_data = pd.read_csv(csv_file_path)
    csv_filtered_dict = {}
    for zeta in zeta_values:
        csv_filtered_dict[zeta] = csv_data[csv_data['zeta_deg'] == zeta].copy()

    return piston_data_dict, csv_filtered_dict


def create_lateral_force_plots(piston_data_dict, csv_data_dict, output_dir="plots"):
    """Create 3 separate Lateral Force plots: Python only, Caspar only, and comparison"""
    zeta_colors = {0: 'red', 5: 'green'}

    # Plot 1: Python only
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
    fig.patch.set_alpha(0)

    for zeta in sorted(csv_data_dict.keys()):
        color = zeta_colors[zeta]
        csv_data = csv_data_dict[zeta]

        if len(csv_data) > 0:
            if 'FSKy' in csv_data.columns and 'FAKy' in csv_data.columns:
                lateral_force_data = csv_data['FSKy'] + csv_data['FAKy']
                ax.plot(csv_data['phi_deg'], lateral_force_data,
                        color=color, linestyle='--',
                        linewidth=PLOT_STYLE['line_width'],
                        label=f'Python (γ={zeta}°)', alpha=1.0)

    apply_plot_style(ax, 'Phi [°]', 'Lateral Force [N]', 'Lateral Force - Analytical')
    create_legend(ax)
    save_plot(fig, output_dir, "Lateral_Force_python_only.png")

    # Plot 2: Caspar only
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
    fig.patch.set_alpha(0)

    for zeta in sorted(piston_data_dict.keys()):
        color = zeta_colors[zeta]
        piston_data = piston_data_dict[zeta]

        if len(piston_data) > 0:
            if 'FSKy' in piston_data.columns and 'FAKy' in piston_data.columns:
                lateral_force_data = piston_data['FSKy'] + piston_data['FAKy']
                ax.plot(piston_data['phi_deg_piston'], lateral_force_data,
                        color=color, linestyle='-',
                        linewidth=PLOT_STYLE['line_width'],
                        label=f'Caspar (γ={zeta}°)', alpha=1.0)

    apply_plot_style(ax, 'Phi [°]', 'Lateral Force [N]', 'Lateral Force - Numerical')
    create_legend(ax)
    save_plot(fig, output_dir, "Lateral_Force_caspar_only.png")

    # Plot 3: Comparison (Python vs Caspar)
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
    fig.patch.set_alpha(0)

    for zeta in sorted(piston_data_dict.keys()):
        color = zeta_colors[zeta]

        # Plot Caspar data
        piston_data = piston_data_dict[zeta]
        if len(piston_data) > 0:
            if 'FSKy' in piston_data.columns and 'FAKy' in piston_data.columns:
                lateral_force_data = piston_data['FSKy'] + piston_data['FAKy']
                ax.plot(piston_data['phi_deg_piston'], lateral_force_data,
                        color=color, linestyle='-',
                        linewidth=PLOT_STYLE['line_width'],
                        label=f'Caspar (γ={zeta}°)', alpha=1.0)

        # Plot Python data
        csv_data = csv_data_dict[zeta]
        if len(csv_data) > 0:
            if 'FSKy' in csv_data.columns and 'FAKy' in csv_data.columns:
                lateral_force_data = csv_data['FSKy'] + csv_data['FAKy']
                ax.plot(csv_data['phi_deg'], lateral_force_data,
                        color=color, linestyle='--',
                        linewidth=PLOT_STYLE['line_width'],
                        label=f'Python (γ={zeta}°)', alpha=1.0)

    apply_plot_style(ax, 'Phi [°]', 'Lateral Force [N]', 'Lateral Force - Analytical vs Numerical')
    create_legend(ax)
    save_plot(fig, output_dir, "Lateral_Force_comparison.png")


def create_plots(piston_data_dict, csv_data_dict, force_columns, output_dir="plots"):
    """Create and save individual plots (excluding Querkraft)"""
    zeta_colors = {0: 'red', 5: 'green'}

    # Filter out Querkraft since it's handled separately
    force_columns_filtered = [col for col in force_columns if col != 'Querkraft']

    # Plot configuration mapping
    plot_config = {
        'FAK_inclined': {
            'ylabel': 'FSK [N]',
            'title': 'Total Axial Force',
            'filename': 'FSK_comparison.png'
        },
        'FSKy': {
            'ylabel': 'FSKy [N]',
            'title': 'Radial Reaction Force',
            'filename': 'FSKy_comparison.png'
        },
        'FAKy': {
            'ylabel': 'FAKy [N]',
            'title': 'Radial Piston Force',
            'filename': 'FAKy_comparison.png'
        }
    }

    for force_col in force_columns_filtered:
        fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])
        fig.patch.set_alpha(0)

        for zeta in sorted(piston_data_dict.keys()):
            color = zeta_colors[zeta]

            # Plot piston data
            piston_data = piston_data_dict[zeta]
            if len(piston_data) > 0:
                # Create FSK from FAK_inclined if this is FAK_inclined column
                if force_col == 'FAK_inclined':
                    if 'FAK_inclined' in piston_data.columns:
                        # Calculate FSK = FAK_inclined * cos(gamma) / cos(14°)
                        gamma_rad = np.radians(zeta)
                        cos_14_deg = np.cos(np.radians(14))
                        fsk_data = piston_data['FAK_inclined'] * np.cos(gamma_rad) / cos_14_deg
                        ax.plot(piston_data['phi_deg_piston'], fsk_data,
                                color=color, linestyle='-',
                                linewidth=PLOT_STYLE['line_width'],
                                label=f'Caspar (γ={zeta}°)', alpha=1.0)
                # Plot regular columns
                elif force_col in piston_data.columns:
                    ax.plot(piston_data['phi_deg_piston'], piston_data[force_col],
                            color=color, linestyle='-',
                            linewidth=PLOT_STYLE['line_width'],
                            label=f'Caspar (γ={zeta}°)', alpha=1.0)

            # Plot CSV data
            csv_data = csv_data_dict[zeta]
            if len(csv_data) > 0:
                # Create FSK from FAK_inclined if this is FAK_inclined column
                if force_col == 'FAK_inclined':
                    if 'FAK_inclined' in csv_data.columns:
                        # Calculate FSK = FAK_inclined * cos(gamma) / cos(14°)
                        gamma_rad = np.radians(zeta)
                        cos_14_deg = np.cos(np.radians(14))
                        fsk_data = csv_data['FAK_inclined'] * np.cos(gamma_rad) / cos_14_deg
                        ax.plot(csv_data['phi_deg'], fsk_data,
                                color=color, linestyle='--',
                                linewidth=PLOT_STYLE['line_width'],
                                label=f'Python (γ={zeta}°)', alpha=1.0)
                # Plot regular columns
                elif force_col in csv_data.columns:
                    ax.plot(csv_data['phi_deg'], csv_data[force_col],
                            color=color, linestyle='--',
                            linewidth=PLOT_STYLE['line_width'],
                            label=f'Python (γ={zeta}°)', alpha=1.0)

        # Get plot configuration or use defaults
        config = plot_config.get(force_col, {
            'ylabel': f'{force_col} [N]',
            'title': f'{force_col} Comparison',
            'filename': f"{force_col}_comparison.png"
        })

        apply_plot_style(ax, 'Phi [°]', config['ylabel'], config['title'])
        create_legend(ax)
        save_plot(fig, output_dir, config['filename'])


def main():
    # File paths
    piston_folder_path = "C:/Users/MIT/Desktop/thesis_temp/Results_inclined_piston_forces_caspar/Results/Friction"
    csv_file_path = "piston_forces_multiple_zeta.csv"

    # Force columns (Querkraft will be handled separately)
    force_columns = ['FAK_inclined', 'FSKy', 'FAKy', 'Querkraft']

    # Load data
    piston_data_dict, csv_data_dict = load_data(piston_folder_path, csv_file_path)

    # Create regular plots (excluding Querkraft)
    create_plots(piston_data_dict, csv_data_dict, force_columns)

    # Create the 3 separate Lateral Force plots
    create_lateral_force_plots(piston_data_dict, csv_data_dict)

    print("Done! Created plots:")
    print("- FSK_comparison.png")
    print("- FSKy_comparison.png")
    print("- FAKy_comparison.png")
    print("- Lateral_Force_python_only.png")
    print("- Lateral_Force_caspar_only.png")
    print("- Lateral_Force_comparison.png")


if __name__ == "__main__":
    main()