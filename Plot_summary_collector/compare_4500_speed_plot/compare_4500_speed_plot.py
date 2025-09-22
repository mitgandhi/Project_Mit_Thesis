# plot_4500_rpm_multi_comparison.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def parse_operating_conditions(filepath):
    """Extract speed from operating conditions file."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('speed'):
                    speed = float(line.split()[1])
                    return speed
    except Exception as e:
        print(f"Error reading operating conditions: {e}")
    return None


def find_4500_rpm_runs(base_folder, tolerance=100):
    """Find all runs with speed close to 4500 RPM in a base folder."""
    target_speed = 4500
    matching_runs = []

    if not os.path.isdir(base_folder):
        print(f"Warning: {base_folder} not found")
        return matching_runs

    subdirs = [d for d in os.listdir(base_folder)
               if os.path.isdir(os.path.join(base_folder, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(base_folder, subdir)
        op_file = os.path.join(subdir_path, 'input', 'operatingconditions.txt')

        speed = parse_operating_conditions(op_file)
        if speed and abs(speed - target_speed) <= tolerance:
            matching_runs.append({
                'label': subdir,
                'speed': speed,
                'filepath': subdir_path
            })
            print(f"Found 4500 RPM run: {subdir} (Speed: {speed:.0f} RPM)")

    return matching_runs


def scan_available_mat_files(run_data):
    """Scan and report all available MAT files in a run directory."""
    print(f"\n--- Scanning MAT files for {run_data['label']} ---")

    plots_dir = os.path.join(run_data['filepath'], 'output', 'piston', 'Plots')
    if not os.path.exists(plots_dir):
        print(f"  ❌ Plots directory not found: {plots_dir}")
        return []

    mat_files = []
    for root, dirs, files in os.walk(plots_dir):
        for file in files:
            if file.endswith('.mat'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, run_data['filepath'])
                mat_files.append((rel_path, full_path))
                print(f"  ✓ Found: {rel_path}")

    if not mat_files:
        print(f"  ❌ No MAT files found in {plots_dir}")

    return mat_files


def load_data_from_mat_files(run_data):
    """Load all available data from MAT files for a run."""
    data = {}

    # First, scan what's available
    available_files = scan_available_mat_files(run_data)

    # Define possible file paths and their corresponding data types
    file_mappings = [
        # Deformation data
        ('Piston_Pressure_Deformation/deformation_data.mat', 'deformation', 'comparisonData'),

        # Contact pressure data - try multiple possible locations
        ('Piston_Contact_Pressure/contact_pressure_data.mat', 'contact_pressure', 'comparisonData'),
        ('Piston_Contact_Pressure_cumulative/contact_pressure_data.mat', 'cumulative_contact', 'comparisonData'),
        ('Piston_Contact_Pressure_cummulativ_grid/cumulative_frame.mat', 'cumulative_grid', None),

        # Gap height data
        ('Piston_Gap_Height/gap_height_data.mat', 'gap_height', 'gapSummary'),
    ]

    for rel_path, data_type, mat_key in file_mappings:
        full_path = os.path.join(run_data['filepath'], 'output', 'piston', 'Plots', rel_path)

        if os.path.exists(full_path):
            try:
                print(f"  📂 Loading: {rel_path}")
                mat_data = loadmat(full_path)
                print(f"    Available keys: {list(mat_data.keys())}")

                if data_type == 'deformation' and mat_key in mat_data:
                    comparison_data = mat_data[mat_key]
                    data['deformation'] = {
                        'max_deformation': comparison_data['maxDeformation'][0][0].flatten(),
                        'degrees': comparison_data['degrees'][0][0].flatten(),
                        'max_value': float(comparison_data['maxValue'][0][0].item()),
                        'phi_max': float(comparison_data['phiMax'][0][0].item())
                    }
                    print(f"    ✓ Deformation data loaded successfully")

                elif data_type == 'contact_pressure' and mat_key in mat_data:
                    comparison_data = mat_data[mat_key]
                    print(f"    Contact pressure data keys: {[key for key in comparison_data.dtype.names]}")

                    # Check what fields are actually available
                    available_fields = comparison_data.dtype.names

                    # Try to find the correct field names
                    cumulative_field = None
                    mean_field = None
                    max_field = None
                    phi_max_field = None
                    degrees_field = None

                    for field in available_fields:
                        field_lower = field.lower()
                        if 'cumulative' in field_lower and 'pressure' in field_lower:
                            # Check if it's an array or scalar
                            field_data = comparison_data[field][0][0]
                            if hasattr(field_data, '__len__') and len(field_data) > 1:
                                cumulative_field = field
                        elif 'mean' in field_lower and 'pressure' in field_lower:
                            mean_field = field
                        elif field_lower == 'maxvalue':
                            max_field = field
                        elif field_lower == 'phimax':
                            phi_max_field = field
                        elif field_lower == 'degrees':
                            degrees_field = field
                        elif 'contact' in field_lower and 'pressure' in field_lower:
                            if mean_field is None:
                                mean_field = field

                    print(
                        f"    Found fields - cumulative: {cumulative_field}, mean: {mean_field}, max: {max_field}, phi_max: {phi_max_field}, degrees: {degrees_field}")

                    if degrees_field and mean_field:
                        contact_data = {
                            'degrees': comparison_data[degrees_field][0][0].flatten(),
                            'mean_pressure': comparison_data[mean_field][0][0].flatten(),
                        }

                        # For cumulative pressure, use mean pressure data if no proper cumulative array exists
                        if cumulative_field:
                            contact_data['cumulative_pressure'] = comparison_data[cumulative_field][0][0].flatten()
                        else:
                            # Use mean pressure data for cumulative plots as fallback
                            contact_data['cumulative_pressure'] = comparison_data[mean_field][0][0].flatten()
                            print(f"    ℹ️  Using mean pressure data for cumulative plots (no cumulative array found)")

                        if max_field:
                            contact_data['max_value'] = float(comparison_data[max_field][0][0].item())
                        else:
                            # Calculate max from available data
                            contact_data['max_value'] = float(np.max(contact_data['mean_pressure']))

                        if phi_max_field:
                            contact_data['phi_max'] = float(comparison_data[phi_max_field][0][0].item())
                        else:
                            # Find angle of maximum value
                            max_idx = np.argmax(contact_data['mean_pressure'])
                            contact_data['phi_max'] = float(contact_data['degrees'][max_idx])

                        data['contact_pressure'] = contact_data
                        print(f"    ✓ Contact pressure data loaded successfully")
                    else:
                        print(f"    ⚠️  Required fields not found in contact pressure data")

                elif data_type == 'gap_height' and mat_key in mat_data:
                    gap_summary = mat_data[mat_key]
                    data['gap_height'] = {
                        'mean_gap': gap_summary['meanGap'][0][0].flatten(),
                        'degrees': gap_summary['degrees'][0][0].flatten(),
                        'max_value': float(gap_summary['maxValue'][0][0].item()),
                        'phi_max': float(gap_summary['phiMax'][0][0].item())
                    }
                    print(f"    ✓ Gap height data loaded successfully")

                elif data_type == 'cumulative_grid':
                    # Handle cumulative grid data differently
                    print(f"    ℹ️  Found cumulative grid data (different format)")

                else:
                    if mat_key:
                        print(f"    ⚠️  Expected key '{mat_key}' not found in {rel_path}")
                    print(f"    Available keys: {list(mat_data.keys())}")

            except Exception as e:
                print(f"    ❌ Error loading {rel_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ❌ Not found: {rel_path}")

    return data


def smart_annotation_position(phi_max, max_value, plot_type="default"):
    """Calculate smart positioning for annotations to keep them within plot bounds."""
    if phi_max > 300:
        text_x = phi_max - 60
        text_y = max_value + max_value * 0.05
    elif phi_max < 60:
        text_x = phi_max + 60
        text_y = max_value + max_value * 0.05
    else:
        text_x = phi_max + 30
        text_y = max_value + max_value * 0.05

    return text_x, text_y


def create_comparison_plot(plot_type, all_run_data, folder_labels, colors, output_dir):
    """Create a comparison plot for a specific parameter type."""

    plt.figure(figsize=(14, 8))

    plot_configs = {
        'deformation': {
            'title': 'Maximum Deformation vs. Shaft Angle at 4500 RPM',
            'ylabel': 'Maximum Deformation [μm]',
            'data_key': 'max_deformation',
            'unit': 'μm'
        },
        'cumulative_pressure': {
            'title': 'Cumulative Contact Pressure vs. Shaft Angle at 4500 RPM',
            'ylabel': 'Cumulative Contact Pressure [Pa·m²]',
            'data_key': 'cumulative_pressure',
            'unit': 'Pa·m²'
        },
        'mean_pressure': {
            'title': 'Mean Contact Pressure vs. Shaft Angle at 4500 RPM',
            'ylabel': 'Mean Contact Pressure [Pa]',
            'data_key': 'mean_pressure',
            'unit': 'Pa'
        },
        'mean_gap': {
            'title': 'Mean Gap Height vs. Shaft Angle at 4500 RPM',
            'ylabel': 'Mean Gap Height [μm]',
            'data_key': 'mean_gap',
            'unit': 'μm'
        }
    }

    config = plot_configs[plot_type]
    data_plotted = False

    for i, (folder_label, runs_data) in enumerate(zip(folder_labels, all_run_data)):
        for j, (run_data, data) in enumerate(runs_data):

            # Determine which data source to use
            data_source = None
            if plot_type == 'deformation' and 'deformation' in data:
                data_source = data['deformation']
            elif plot_type in ['cumulative_pressure', 'mean_pressure']:
                # For pressure plots, use contact_pressure data
                if 'contact_pressure' in data:
                    data_source = data['contact_pressure']
                    print(f"  📊 Using contact_pressure data for {plot_type}")
                elif 'cumulative_contact' in data:
                    data_source = data['cumulative_contact']
                    print(f"  📊 Using cumulative_contact data for {plot_type}")
            elif plot_type == 'mean_gap' and 'gap_height' in data:
                data_source = data['gap_height']

            if data_source is None:
                print(f"  ❌ No {plot_type} data found for {run_data['label']}")
                continue

            if config['data_key'] not in data_source:
                print(f"  ⚠️  {config['data_key']} not found in data for {run_data['label']}")
                print(f"      Available keys: {list(data_source.keys())}")
                continue

            degrees = data_source['degrees']
            plot_data = data_source[config['data_key']]
            max_value = data_source['max_value']
            phi_max = data_source['phi_max']

            # Create label for this run
            run_label = f"{folder_label} - {run_data['label']}"
            run_label += f" (Speed: {run_data['speed']:.0f} RPM)"

            # Plot the data
            line_style = '-' if j == 0 else '--'
            plt.plot(degrees, plot_data,
                     color=colors[i],
                     linestyle=line_style,
                     linewidth=3,
                     label=run_label,
                     alpha=0.8)

            # Add annotation for peak value with smart positioning
            text_x, text_y = smart_annotation_position(phi_max, max_value, plot_type)

            plt.annotate(f'Peak: {max_value:.1f} {config["unit"]}\n@ {phi_max:.0f}°',
                         xy=(phi_max, max_value),
                         xytext=(text_x, text_y),
                         arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5),
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.2),
                         color=colors[i],
                         fontweight='bold',
                         ha='center')

            print(f"  ✓ Plotted: {run_data['label']} (Peak: {max_value:.1f} {config['unit']} @ {phi_max:.0f}°)")
            data_plotted = True

    if not data_plotted:
        print(f"❌ No {plot_type} data found for plotting!")
        plt.close()
        return False

    # Format the plot
    plt.title(f"{config['title']}\nComparison: Optimal vs Test Clearance",
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Shaft Angle [°]', fontsize=14, fontweight='bold')
    plt.ylabel(config['ylabel'], fontsize=14, fontweight='bold')

    # Customize grid and axes
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, 360)

    # Set y-axis limits with some padding for annotations
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max * 1.15)  # Add 15% padding at top for annotations

    plt.xticks(np.arange(0, 361, 60), fontsize=12)
    plt.yticks(fontsize=12)

    # Add legend
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9,
               shadow=True, fancybox=True)

    # Add professional styling
    plt.tight_layout()

    # Save the plot
    filename_map = {
        'deformation': 'Max_Deformation_4500RPM_Comparison',
        'cumulative_pressure': 'Cumulative_Contact_Pressure_4500RPM_Comparison',
        'mean_pressure': 'Mean_Contact_Pressure_4500RPM_Comparison',
        'mean_gap': 'Mean_Gap_Height_4500RPM_Comparison'
    }

    filename = filename_map[plot_type]

    # Save PNG and PDF
    png_file = os.path.join(output_dir, f'{filename}.png')
    pdf_file = os.path.join(output_dir, f'{filename}.pdf')

    plt.savefig(png_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(pdf_file, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    plt.show()

    print(f"  📁 Saved: {filename}.png and {filename}.pdf")
    return True


def create_all_4500_rpm_comparison_plots():
    """Create comparison plots for all parameters at 4500 RPM."""

    # Define the two base folders
    base_folders = [
        r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T08_320dp_100d_dZ_19.415_dK_19.387',
        r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T05_variable speed_ speedK_1',
        r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run7_ReDimension_L57\T01_optimized_clearance'
    ]

    # Custom labels for the plot
    folder_labels = [
        "Optimal Clearance",
        "Measurement",
        "Optimal Clearance shorter Piston"
    ]

    # Colors for each configuration
    colors = ['blue', 'red', 'green']

    print("=" * 70)
    print("CREATING 4500 RPM MULTI-PARAMETER COMPARISON PLOTS")
    print("=" * 70)

    # Create output directory
    output_dir = '4500_RPM_Presentation_Plots'
    os.makedirs(output_dir, exist_ok=True)

    # Collect data from both folders
    all_run_data = []

    for i, (base_folder, folder_label) in enumerate(zip(base_folders, folder_labels)):
        print(f"\n{'=' * 50}")
        print(f"Searching in: {folder_label}")
        print(f"Path: {base_folder}")
        print(f"{'=' * 50}")

        # Find 4500 RPM runs in this folder
        matching_runs = find_4500_rpm_runs(base_folder)

        if not matching_runs:
            print(f"  ❌ No 4500 RPM runs found in {folder_label}")
            all_run_data.append([])
            continue

        # Load data for each matching run
        folder_data = []
        for run_data in matching_runs:
            print(f"\n📂 Processing run: {run_data['label']}")
            data = load_data_from_mat_files(run_data)
            folder_data.append((run_data, data))

            # Print summary of loaded data
            loaded_types = list(data.keys())
            if loaded_types:
                print(f"  ✅ Successfully loaded: {', '.join(loaded_types)}")
            else:
                print(f"  ⚠️  No data could be loaded for this run")

        all_run_data.append(folder_data)

    # Check if any data was found
    total_runs = sum(len(folder_data) for folder_data in all_run_data)
    if total_runs == 0:
        print("\n❌ No 4500 RPM data found!")
        return

    print(f"\n{'=' * 70}")
    print(f"DATA COLLECTION SUMMARY")
    print(f"{'=' * 70}")
    for i, (folder_label, folder_data) in enumerate(zip(folder_labels, all_run_data)):
        print(f"{folder_label}: {len(folder_data)} runs found")

    # Create plots for each parameter
    plot_types = ['deformation', 'cumulative_pressure', 'mean_pressure', 'mean_gap']
    successful_plots = 0

    for plot_type in plot_types:
        print(f"\n{'=' * 50}")
        print(f"Creating {plot_type.replace('_', ' ').title()} comparison plot...")
        print(f"{'=' * 50}")

        success = create_comparison_plot(plot_type, all_run_data, folder_labels, colors, output_dir)
        if success:
            successful_plots += 1

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"✅ Successfully created {successful_plots}/{len(plot_types)} comparison plots")
    print(f"📁 Output directory: {output_dir}")

    if successful_plots > 0:
        print(f"🎯 Generated plots:")
        plot_files = [
            'Max_Deformation_4500RPM_Comparison',
            'Cumulative_Contact_Pressure_4500RPM_Comparison',
            'Mean_Contact_Pressure_4500RPM_Comparison',
            'Mean_Gap_Height_4500RPM_Comparison'
        ]
        for i, plot_type in enumerate(plot_types):
            if i < successful_plots:
                print(f"   • {plot_files[i]}.png/.pdf")

    if successful_plots < len(plot_types):
        print(f"⚠️  Note: {len(plot_types) - successful_plots} plots could not be created due to missing data")
        print(f"   Check the scan results above to see what MAT files are available")

    print(f"\n🎯 All available plots are ready for your presentation!")


if __name__ == "__main__":
    create_all_4500_rpm_comparison_plots()