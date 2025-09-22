# run_unified_batch_parallel.py - Enhanced version with full multiprocessing utilization and consistent color scaling

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil
from scipy.io import loadmat
try:
    from .unified_piston_plots import generate_piston_plots
    from .contact_pressure_superimposed import create_cumulative_pressure_map
    from .contact_pressure_cummulative_grid import piston_contact_pressure
except (ImportError, ValueError):
    from unified_piston_plots import generate_piston_plots  # type: ignore
    from contact_pressure_superimposed import create_cumulative_pressure_map  # type: ignore
    from contact_pressure_cummulative_grid import piston_contact_pressure  # type: ignore
import multiprocessing as mp
from multiprocessing import cpu_count
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob




def get_optimal_process_count():
    """Get optimal number of processes to maximize CPU utilization."""
    cpu_cores = cpu_count()
    # Use all available cores
    return cpu_cores


def setup_matplotlib_for_multiprocessing():
    """Configure matplotlib for multiprocessing environment."""
    plt.switch_backend('Agg')  # Use non-interactive backend
    plt.ioff()  # Turn off interactive mode


def clean_plot_directories(base_folders):
    """
    Clean existing plot directories to ensure fresh plots with consistent color scales.

    Parameters:
    -----------
    base_folders : list
        List of base folder paths to clean
    """
    print(f"\n=== CLEANING EXISTING PLOT DIRECTORIES ===")

    directories_to_clean = [
        'Piston_Contact_Pressure_cumulative',
        'Piston_Contact_Pressure_cummulativ_grid'
    ]

    total_cleaned = 0

    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            continue

        print(f"Cleaning plots in: {base_folder}")

        # Get all simulation directories
        subdirs = [d for d in os.listdir(base_folder)
                   if os.path.isdir(os.path.join(base_folder, d))]

        for subdir in subdirs:
            subdir_path = os.path.join(base_folder, subdir)
            plots_base_path = os.path.join(subdir_path, 'output', 'piston', 'Plots')

            if os.path.exists(plots_base_path):
                for dir_to_clean in directories_to_clean:
                    dir_path = os.path.join(plots_base_path, dir_to_clean)
                    if os.path.exists(dir_path):
                        try:
                            shutil.rmtree(dir_path)
                            print(f"  ✔ Removed: {os.path.basename(subdir)}/{dir_to_clean}")
                            total_cleaned += 1
                        except Exception as e:
                            print(f"  ⚠ Failed to remove {dir_path}: {e}")

    print(f"Total directories cleaned: {total_cleaned}")


def get_global_color_scale(base_folders):
    """
    Analyze all existing data to determine global min/max values for consistent color scaling.

    Parameters:
    -----------
    base_folders : list
        List of base folder paths

    Returns:
    --------
    tuple : (global_min, global_max) for color scaling
    """
    print(f"\n=== ANALYZING DATA FOR GLOBAL COLOR SCALE ===")

    all_values = []

    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            continue

        subdirs = [d for d in os.listdir(base_folder)
                   if os.path.isdir(os.path.join(base_folder, d))]

        for subdir in subdirs:
            subdir_path = os.path.join(base_folder, subdir)

            # Check for existing .mat files with pressure data
            mat_files = glob.glob(os.path.join(subdir_path, 'output', 'piston', 'Plots',
                                               'Piston_Contact_Pressure_cumulative', '*.mat'))

            for mat_file in mat_files:
                try:
                    mat_data = loadmat(mat_file)
                    if 'comparisonData' in mat_data:
                        pressure_data = mat_data['comparisonData']['cumulativeContactPressure'][0][0]
                        all_values.extend(pressure_data.flatten())
                except Exception as e:
                    continue

            # Also check CSV files for additional data
            csv_files = glob.glob(os.path.join(subdir_path, 'output', 'piston', 'Plots',
                                               'Piston_Contact_Pressure_cummulativ_grid', '*.csv'))

            for csv_file in csv_files:
                try:
                    with open(csv_file, 'r') as f:
                        reader = csv.reader(f)
                        rows = list(reader)

                    for row in rows:
                        if len(row) >= 3:
                            try:
                                # Try to extract numerical values
                                val1 = float(row[1])  # Mean pressure
                                val2 = float(row[2])  # Max pressure
                                all_values.extend([val1, val2])
                            except (ValueError, IndexError):
                                continue
                except Exception as e:
                    continue

    if all_values:
        global_min = np.min(all_values)
        global_max = np.max(all_values)
        print(f"Global color scale determined: [{global_min:.2e}, {global_max:.2e}]")
        return global_min, global_max
    else:
        # Default fallback values
        print("No existing data found, using default color scale")
        return 0, 1e6


# === Helpers to parse per‐run inputs ===
def parse_geometry(filepath):
    """Extract gap length (lF) from geometry file."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('lF'):
                    return float(line.split()[1])
    except Exception as e:
        print(f"Error reading geometry file: {e}")
    return None


def parse_operating_conditions(filepath):
    """Extract speed and pressure from operating conditions file."""
    speed = hp = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('speed'):
                    speed = float(line.split()[1])
                elif line.strip().startswith('HP'):
                    hp = float(line.split()[1])
    except Exception as e:
        print(f"Error reading operating conditions file: {e}")
    return speed, hp


def process_single_run(args):
    """
    Process a single simulation run - designed for parallel execution.

    Parameters:
    -----------
    args : tuple
        (run_data, plot_types, plots, degstep, ignore_base, minplot, maxplot, run_enhanced_analysis, global_color_scale)
    """
    run_data, plot_types, plots, degstep, ignore_base, minplot, maxplot, run_enhanced_analysis, global_color_scale = args

    # Setup matplotlib for this process
    setup_matplotlib_for_multiprocessing()

    try:
        process_id = os.getpid()
        print(f"[PID {process_id}] Processing run: {run_data['label']}")

        # 1. Generate standard plots using unified function
        results = generate_piston_plots(
            filepath=run_data['filepath'],
            plot_type=plot_types,
            m=run_data['m'],
            lF=run_data['lF'],
            n=run_data['n'],
            deltap=run_data['deltap'],
            plots=plots,
            degstep=degstep,
            ignore_base=ignore_base,
            minplot=minplot,
            maxplot=maxplot,
            offset=run_data['offset']
        )

        status = "COMPLETED" if results else "NO_PLOTS"
        print(f"[PID {process_id}] Standard plots for {run_data['label']}: {status}")

        # 2. Run enhanced contact pressure analysis if requested
        if run_enhanced_analysis and 'contact_pressure' in str(plot_types).lower():
            try:
                run_enhanced_contact_pressure_analysis(run_data, global_color_scale)
                print(f"[PID {process_id}] Enhanced analysis completed for {run_data['label']}")
                enhanced_status = "COMPLETED"
            except Exception as e:
                print(f"[PID {process_id}] Enhanced analysis failed for {run_data['label']}: {e}")
                enhanced_status = "FAILED"
        else:
            enhanced_status = "SKIPPED"

        return {
            'label': run_data['label'],
            'filepath': run_data['filepath'],
            'standard_plots': status,
            'enhanced_analysis': enhanced_status,
            'process_id': process_id,
            'success': True
        }

    except Exception as e:
        print(f"[PID {process_id}] ERROR processing {run_data['label']}: {e}")
        return {
            'label': run_data['label'],
            'filepath': run_data['filepath'],
            'error': str(e),
            'process_id': process_id,
            'success': False
        }


def run_enhanced_contact_pressure_analysis(run_data, global_color_scale=None):
    """
    Run enhanced contact pressure analysis including cumulative grids and superimposed maps.
    Optimized for parallel execution with consistent color scaling.
    """
    # Prepare kwargs for the enhanced analysis functions
    kwargs = {
        "filepath": run_data['filepath'],
        "m": run_data['m'],
        "lF": run_data['lF'],
        "n": run_data['n'],
        "deltap": run_data['deltap'],
        "plots": 360,
        "degstep": 1,
        "ignore": 0,
        "ognore": 0,
        "minplot": 0,
        "maxplot": 0,
        "offset": run_data.get('offset', 0)
    }

    # Add global color scale if provided
    if global_color_scale is not None:
        kwargs['global_vmin'] = global_color_scale[0]
        kwargs['global_vmax'] = global_color_scale[1]

    # Check if cumulative grid analysis already exists
    cumulative_grid_path = os.path.join(
        run_data['filepath'],
        'output', 'piston', 'Plots',
        'Piston_Contact_Pressure_cummulativ_grid',
        'Cumulative_Pressure_Stats.csv'
    )

    # Always run analysis since we cleaned directories
    piston_contact_pressure(**kwargs)

    # Check if superimposed analysis already exists
    superimposed_path = os.path.join(
        run_data['filepath'],
        'output', 'piston', 'Plots',
        'Piston_Contact_Pressure_cumulative',
        'contact_pressure_data.mat'
    )

    # Always run analysis since we cleaned directories
    create_cumulative_pressure_map(**kwargs)


def create_consistent_heatmaps(base_folders, global_color_scale):
    """
    Create heatmaps with consistent color scaling across all runs.

    Parameters:
    -----------
    base_folders : list
        List of base folder paths
    global_color_scale : tuple
        (min_value, max_value) for consistent color scaling
    """
    print(f"\n=== CREATING CONSISTENT HEATMAPS ===")
    print(f"Using global color scale: [{global_color_scale[0]:.2e}, {global_color_scale[1]:.2e}]")

    setup_matplotlib_for_multiprocessing()

    total_heatmaps = 0

    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            continue

        subdirs = [d for d in os.listdir(base_folder)
                   if os.path.isdir(os.path.join(base_folder, d))]

        for subdir in subdirs:
            subdir_path = os.path.join(base_folder, subdir)

            # Look for cumulative pressure data
            mat_path = os.path.join(
                subdir_path, 'output', 'piston', 'Plots',
                'Piston_Contact_Pressure_cumulative', 'contact_pressure_data.mat'
            )

            if os.path.exists(mat_path):
                try:
                    # Load data
                    mat_data = loadmat(mat_path)
                    pressure_data = mat_data['comparisonData']['cumulativeContactPressure'][0][0]

                    # Create heatmap with consistent color scale
                    plt.figure(figsize=(12, 8))

                    im = plt.imshow(pressure_data,
                                    cmap='jet',
                                    aspect='auto',
                                    vmin=global_color_scale[0],
                                    vmax=global_color_scale[1])

                    plt.colorbar(im, label='Cumulative Contact Pressure [Pa]')
                    plt.title(f'Cumulative Contact Pressure Heatmap - {subdir}')
                    plt.xlabel('Circumferential Position')
                    plt.ylabel('Axial Position')

                    # Save with consistent naming
                    output_dir = os.path.join(
                        subdir_path, 'output', 'piston', 'Plots',
                        'Piston_Contact_Pressure_cumulative'
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    heatmap_path = os.path.join(output_dir, 'Cumulative_Contact_Pressure_Heatmap.png')
                    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    total_heatmaps += 1
                    print(f"  ✔ Created consistent heatmap for {subdir}")

                except Exception as e:
                    print(f"  ⚠ Failed to create heatmap for {subdir}: {e}")

    print(f"Total consistent heatmaps created: {total_heatmaps}")


def parallel_run_processing(all_runs_data, plot_types='all', run_enhanced_analysis=True, global_color_scale=None):
    """
    Process all runs in parallel using all available CPU cores.

    Parameters:
    -----------
    all_runs_data : list
        List of run data dictionaries
    plot_types : str or list
        Plot types to generate
    run_enhanced_analysis : bool
        Whether to run enhanced analysis
    global_color_scale : tuple
        Global color scale for consistent plotting

    Returns:
    --------
    list : Processing results for each run
    """
    # Default parameters
    plots = 360
    degstep = 1
    ignore_base = 0
    minplot = 0
    maxplot = 0

    # Get optimal number of processes
    num_processes = get_optimal_process_count()
    print(f"\n=== PARALLEL PROCESSING SETUP ===")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Using processes: {num_processes}")
    print(f"Total runs to process: {len(all_runs_data)}")
    print(f"Expected CPU utilization: ~100%")
    print(f"Global color scale: {global_color_scale}")

    # Prepare arguments for each run
    run_args = []
    for run_data in all_runs_data:
        args = (
            run_data, plot_types, plots, degstep, ignore_base,
            minplot, maxplot, run_enhanced_analysis, global_color_scale
        )
        run_args.append(args)

    # Process runs in parallel
    print(f"\n=== STARTING PARALLEL EXECUTION ===")
    start_time = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all jobs
        future_to_run = {executor.submit(process_single_run, args): args[0]['label']
                         for args in run_args}

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_run):
            run_label = future_to_run[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if result['success']:
                    print(f"✅ [{completed}/{len(all_runs_data)}] Completed: {run_label}")
                else:
                    print(f"❌ [{completed}/{len(all_runs_data)}] Failed: {run_label}")

            except Exception as e:
                print(f"❌ [{completed + 1}/{len(all_runs_data)}] Exception for {run_label}: {e}")
                results.append({
                    'label': run_label,
                    'error': str(e),
                    'success': False
                })
                completed += 1

    end_time = time.time()
    total_time = end_time - start_time

    # Print summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\n=== PARALLEL PROCESSING SUMMARY ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per run: {total_time / len(all_runs_data):.2f} seconds")
    print(f"Successful runs: {successful}")
    print(f"Failed runs: {failed}")
    print(f"Processes used: {num_processes}")
    print(f"Efficiency: {len(all_runs_data) * 100 / (total_time * num_processes):.1f}% of theoretical maximum")

    return results


def parallel_csv_data_collection(base_folders, custom_labels):
    """
    Collect CSV data from all base folders in parallel.
    """

    def collect_folder_data(args):
        base_folder, pump_type = args
        setup_matplotlib_for_multiprocessing()

        print(f"[PID {os.getpid()}] Collecting data for {pump_type}")

        if not os.path.isdir(base_folder):
            return pump_type, None

        subdirs = [d for d in os.listdir(base_folder)
                   if os.path.isdir(os.path.join(base_folder, d))]
        filepaths = [os.path.join(base_folder, d) for d in subdirs]

        if not filepaths:
            return pump_type, None

        data = {
            'speeds': [],
            'mean_pressures': [],
            'labels': [],
            'delta_pressures': [],
            'filepaths': filepaths
        }

        for filepath in filepaths:
            label = os.path.basename(filepath)

            # Parse operating conditions
            geom_file = os.path.join(filepath, 'input', 'geometry.txt')
            op_file = os.path.join(filepath, 'input', 'operatingconditions.txt')

            lF_val = parse_geometry(geom_file)
            speed_val, hp_val = parse_operating_conditions(op_file)

            if speed_val is None or hp_val is None:
                continue

            # Try to load mean pressure from superimposed analysis
            mat_path = os.path.join(filepath, 'output', 'piston', 'Plots',
                                    'Piston_Contact_Pressure_cumulative', 'contact_pressure_data.mat')

            if os.path.exists(mat_path):
                try:
                    mat_data = loadmat(mat_path)
                    mean_pressure = np.mean(mat_data['comparisonData']['meanContactPressure'][0][0])

                    data['speeds'].append(speed_val)
                    data['mean_pressures'].append(mean_pressure)
                    data['labels'].append(label)
                    data['delta_pressures'].append(hp_val)

                except Exception as e:
                    print(f"    Error reading mat file for {label}: {e}")

        return pump_type, data

    # Prepare arguments
    folder_args = [(base_folders[i], custom_labels[i]) for i in range(len(base_folders))]

    # Collect data in parallel
    cumulative_mean_data = {}
    with ProcessPoolExecutor(max_workers=min(len(base_folders), get_optimal_process_count())) as executor:
        results = executor.map(collect_folder_data, folder_args)

        for pump_type, data in results:
            if data is not None:
                cumulative_mean_data[pump_type] = data

    return cumulative_mean_data


def create_comparison_plots_parallel(base_folders, custom_labels=None):
    """
    Create comparison plots with parallel data collection.
    """
    print(f"\n=== Creating Comparison Plots (Parallel) ===")

    if custom_labels is None:
        custom_labels = [f"Pump_{i + 1}" for i in range(len(base_folders))]

    # Create output directory for comparison plots
    comparison_output_dir = 'Comparison_Plots'
    os.makedirs(comparison_output_dir, exist_ok=True)

    # Collect data in parallel
    print("Collecting data from all base folders in parallel...")
    cumulative_mean_data = parallel_csv_data_collection(base_folders, custom_labels)

    # Setup matplotlib for main process
    setup_matplotlib_for_multiprocessing()

    # Create plots using the collected data
    create_comparison_plots_from_data(cumulative_mean_data, comparison_output_dir)
    create_csv_profile_plots_parallel(base_folders, custom_labels, comparison_output_dir)

    print(f"✅ All comparison plots saved to {comparison_output_dir}")


def create_comparison_plots_from_data(cumulative_mean_data, output_dir):
    """Create comparison plots from collected data."""

    # === COMBINED MEAN PRESSURE VS SPEED PLOT ===
    plt.figure(figsize=(12, 8))

    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p']

    for idx, (pump_type, data) in enumerate(cumulative_mean_data.items()):
        if data['speeds'] and data['mean_pressures']:
            # Sort by speed
            sorted_indices = np.argsort(data['speeds'])
            sorted_speeds = [data['speeds'][i] for i in sorted_indices]
            sorted_pressures = [data['mean_pressures'][i] for i in sorted_indices]

            plt.plot(sorted_speeds, sorted_pressures,
                     marker=markers[idx % len(markers)],
                     color=colors[idx % len(colors)],
                     linestyle='-',
                     linewidth=2,
                     label=f'{pump_type}')

    plt.title('Mean Contact Pressure vs. Operating Speed for Different Pump Types', fontsize=14)
    plt.xlabel('Speed (RPM)', fontsize=12)
    plt.ylabel('Mean Contact Pressure (Pa)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Mean_Pressure_vs_Speed_Comparison.png'),
                dpi=300, transparent=True)
    plt.close()

    # === INDIVIDUAL PLOTS FOR EACH PUMP TYPE ===
    for pump_type, data in cumulative_mean_data.items():
        if data['speeds'] and data['mean_pressures']:
            plt.figure(figsize=(10, 6))

            # Sort data
            sorted_indices = np.argsort(data['speeds'])
            sorted_speeds = [data['speeds'][i] for i in sorted_indices]
            sorted_pressures = [data['mean_pressures'][i] for i in sorted_indices]
            sorted_labels = [data['labels'][i] for i in sorted_indices]

            plt.plot(sorted_speeds, sorted_pressures, 'bo-', linewidth=2, markersize=8)

            # Add data labels
            for x, y, label in zip(sorted_speeds, sorted_pressures, sorted_labels):
                plt.annotate(label, (x, y), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=9)

            plt.title(f'{pump_type}: Mean Contact Pressure vs. Operating Speed', fontsize=14)
            plt.xlabel('Speed (RPM)', fontsize=12)
            plt.ylabel('Mean Contact Pressure (Pa)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)

            # Add trend line
            if len(sorted_speeds) > 1:
                z = np.polyfit(sorted_speeds, sorted_pressures, 1)
                p = np.poly1d(z)
                plt.plot(sorted_speeds, p(sorted_speeds), "r--", alpha=0.7,
                         label=f"Trend: y={z[0]:.2e}x+{z[1]:.2e}")
                plt.legend(loc='best')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{pump_type}_Mean_Pressure_vs_Speed.png'),
                        dpi=300, transparent=True)
            plt.close()


def create_csv_profile_plots_parallel(base_folders, custom_labels, output_dir):
    """
    Create CSV-based profile plots with parallel data processing.
    """

    def process_csv_data(args):
        base_folder, base_idx = args
        setup_matplotlib_for_multiprocessing()

        if not os.path.isdir(base_folder):
            return base_idx, None, None

        filepaths = [os.path.join(base_folder, d) for d in os.listdir(base_folder)
                     if os.path.isdir(os.path.join(base_folder, d))]

        max_pressure_data = []
        mean_pressure_data = []
        speed_data = []

        for path in filepaths:
            label = os.path.basename(path)
            csv_path = os.path.join(path, 'output', 'piston', 'Plots',
                                    'Piston_Contact_Pressure_cummulativ_grid',
                                    'Cumulative_Pressure_Stats.csv')

            if os.path.exists(csv_path):
                try:
                    with open(csv_path, 'r') as f:
                        reader = csv.reader(f)
                        rows = list(reader)

                    # Find data start
                    data_start_idx = None
                    for idx, row in enumerate(rows):
                        if row and row[0] == 'Gap Length Index':
                            data_start_idx = idx + 1
                            break

                    if data_start_idx is None:
                        continue

                    gap_indices, max_pressures, avg_pressures = [], [], []
                    for row in rows[data_start_idx:]:
                        if len(row) < 3 or not row[0].strip().isdigit():
                            break
                        gap_indices.append(int(row[0]))
                        avg_pressures.append(float(row[1]))
                        max_pressures.append(float(row[2]))

                    max_pressure_data.append((label, gap_indices, max_pressures))
                    mean_pressure_data.append((label, gap_indices, avg_pressures))

                    # Get speed for summary plot
                    op_file = os.path.join(path, 'input', 'operatingconditions.txt')
                    speed, _ = parse_operating_conditions(op_file)
                    if speed is not None:
                        mean_of_profile = np.mean(avg_pressures)
                        speed_data.append((speed, mean_of_profile))

                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        return base_idx, {
            'max_pressure': max_pressure_data,
            'mean_pressure': mean_pressure_data,
            'speed_data': speed_data
        }

    # Process CSV data in parallel
    folder_args = [(base_folders[i], i) for i in range(len(base_folders))]

    all_data = {}
    with ProcessPoolExecutor(max_workers=min(len(base_folders), get_optimal_process_count())) as executor:
        results = executor.map(process_csv_data, folder_args)

        for base_idx, data in results:
            if data is not None:
                all_data[base_idx] = data

    # Create plots using collected data
    create_profile_plots_from_data(all_data, custom_labels, output_dir)


def create_profile_plots_from_data(all_data, custom_labels, output_dir):
    """Create profile plots from collected data."""
    line_styles = ['-', '--', '-.', ':']

    # === MAX PRESSURE PROFILE PLOT ===
    plt.figure(figsize=(10, 6))

    for base_idx, data in all_data.items():
        line_style = line_styles[base_idx % len(line_styles)]
        colors = plt.cm.get_cmap('tab10', len(data['max_pressure']))

        for i, (label, gap_indices, max_pressures) in enumerate(data['max_pressure']):
            plt.plot(gap_indices, max_pressures, label=label,
                     color=colors(i), linestyle=line_style)

    plt.title('Max Pressure Profile across Gap Length')
    plt.xlabel('Gap Length Index')
    plt.ylabel('Max Cumulative Pressure [Pa]')
    plt.grid(True)
    plt.legend(title='Simulation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Combined_Max_Pressure_From_CSV.png'),
                dpi=300, transparent=True)
    plt.close()

    # === MEAN PRESSURE PROFILE PLOT ===
    plt.figure(figsize=(10, 6))

    for base_idx, data in all_data.items():
        line_style = line_styles[base_idx % len(line_styles)]
        colors = plt.cm.get_cmap('tab10', len(data['mean_pressure']))

        for i, (label, gap_indices, avg_pressures) in enumerate(data['mean_pressure']):
            plt.plot(gap_indices, avg_pressures, label=label,
                     color=colors(i), linestyle=line_style)

    plt.title('Mean Pressure Profile across Gap Length')
    plt.xlabel('Gap Length Index')
    plt.ylabel('Mean Cumulative Pressure [Pa]')
    plt.grid(True)
    plt.legend(title='Simulation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Combined_Mean_Pressure_From_CSV.png'),
                dpi=300, transparent=True)
    plt.close()

    # === SUMMARY PLOT: MEAN PRESSURE VS SPEED PER BASE FOLDER ===
    plt.figure(figsize=(10, 6))
    summary_colors = ['red', 'green', 'blue', 'purple', 'orange']

    for base_idx, data in all_data.items():
        if data['speed_data']:
            sorted_data = sorted(data['speed_data'])
            x, y = zip(*sorted_data)
            plt.plot(x, y, label=custom_labels[base_idx],
                     color=summary_colors[base_idx % len(summary_colors)],
                     linewidth=2.5, marker='o')

    plt.title('Mean Pressure vs Speed')
    plt.xlabel('Speed [rpm]')
    plt.ylabel('Mean of Mean Cumulative Pressure [Pa]')
    plt.grid(True)
    plt.legend(title='Pump Series')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'MeanPressure_vs_Speed_PerBaseFolder.png'),
                dpi=300, transparent=True)
    plt.close()


def run_batch_analysis_parallel(base_folders, plot_types=None, custom_labels=None,
                                run_enhanced_analysis=True, create_comparisons=True,
                                clean_existing_plots=True):
    """
    Enhanced batch analysis with full parallel processing utilizing 100% CPU power.
    Includes consistent color scaling and plot directory cleaning.

    Parameters:
    -----------
    base_folders : list
        List of base folders to process
    plot_types : list or str, optional
        List of plot types to generate or 'all' for all types
    custom_labels : list, optional
        Custom labels for base folders for comparison plots
    run_enhanced_analysis : bool
        Whether to run enhanced contact pressure analysis
    create_comparisons : bool
        Whether to create comparison plots
    clean_existing_plots : bool
        Whether to clean existing plot directories for fresh plots
    """
    if plot_types is None:
        plot_types = 'create_comparisons'

    if custom_labels is None:
        custom_labels = [f"Pump_{i + 1}" for i in range(len(base_folders))]

    print(f"\n{'=' * 60}")
    print(f"PARALLEL BATCH ANALYSIS - MAXIMUM CPU UTILIZATION")
    print(f"WITH CONSISTENT COLOR SCALING AND CLEAN PLOTS")
    print(f"{'=' * 60}")
    print(f"CPU Cores Available: {cpu_count()}")
    print(f"Target CPU Utilization: 100%")
    print(f"Base Folders: {len(base_folders)}")
    print(f"Clean Existing Plots: {clean_existing_plots}")

    # Step 0: Clean existing plot directories if requested
    if clean_existing_plots:
        clean_plot_directories(base_folders)

    # Step 1: Determine global color scale for consistent heatmaps
    print(f"\nDetermining global color scale for consistent heatmaps...")
    global_color_scale = get_global_color_scale(base_folders)

    # Step 2: Collect all runs from all base folders
    all_runs_data = []

    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"Warning: {base_folder} not found; skipping.")
            continue

        # Gather all simulation sub-folders
        subdirs = [d for d in os.listdir(base_folder)
                   if os.path.isdir(os.path.join(base_folder, d))]
        filepaths = [os.path.join(base_folder, d) for d in subdirs]
        labels = [os.path.basename(fp) for fp in filepaths]

        if not filepaths:
            print(f"Warning: No runs in {base_folder}; skipping.")
            continue

        # Parse per-run geometry & operating conditions
        for idx, fp in enumerate(filepaths):
            geom_file = os.path.join(fp, 'input', 'geometry.txt')
            op_file = os.path.join(fp, 'input', 'operatingconditions.txt')

            lF_val = parse_geometry(geom_file)
            speed_val, hp_val = parse_operating_conditions(op_file)

            if lF_val is not None and speed_val is not None and hp_val is not None:
                run_data = {
                    'filepath': fp,
                    'label': labels[idx],
                    'lF': lF_val,
                    'n': speed_val,
                    'deltap': hp_val,
                    'm': 50,  # Default value
                    'offset': 0,  # Default value
                    'base_folder': base_folder
                }
                all_runs_data.append(run_data)
            else:
                print(f"Warning: Skipping '{labels[idx]}' (missing geometry or operating conditions)")

    if not all_runs_data:
        print("No valid runs found. Exiting.")
        return

    # Step 3: Process all runs in parallel (maximum CPU utilization)
    print(f"\nTotal simulation runs to process: {len(all_runs_data)}")
    processing_results = parallel_run_processing(
        all_runs_data, plot_types, run_enhanced_analysis, global_color_scale
    )

    # Step 4: Create consistent heatmaps with global color scaling
    if run_enhanced_analysis and 'contact_pressure' in str(plot_types).lower():
        print(f"\n=== POST-PROCESSING: CONSISTENT HEATMAPS ===")
        try:
            create_consistent_heatmaps(base_folders, global_color_scale)
            print(f"✅ Consistent heatmaps created successfully")
        except Exception as e:
            print(f"⚠ Consistent heatmap creation failed: {e}")

    # Step 5: Create comparison plots with parallel data collection
    if create_comparisons:
        print(f"\n=== CREATING COMPARISON PLOTS (PARALLEL) ===")
        try:
            create_comparison_plots_parallel(base_folders, custom_labels)
            print(f"✅ Comparison plots created successfully")
        except Exception as e:
            print(f"⚠ Comparison plots failed: {e}")

    # Step 6: Final summary
    successful_runs = sum(1 for r in processing_results if r['success'])
    failed_runs = len(processing_results) - successful_runs

    print(f"\n{'=' * 60}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total runs processed: {len(processing_results)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"CPU cores utilized: {cpu_count()}")
    print(f"Processing mode: PARALLEL (Maximum CPU utilization)")
    print(f"Color scale consistency: ENABLED")
    print(f"Global color range: [{global_color_scale[0]:.2e}, {global_color_scale[1]:.2e}]")
    print(f"Plot directories cleaned: {clean_existing_plots}")
    print(f"{'=' * 60}")

    if failed_runs > 0:
        print(f"\nFailed runs:")
        for result in processing_results:
            if not result['success']:
                print(f"  - {result['label']}: {result.get('error', 'Unknown error')}")

    # Step 7: Generate summary report
    generate_analysis_summary_report(processing_results, base_folders, custom_labels, global_color_scale)


def generate_analysis_summary_report(processing_results, base_folders, custom_labels, global_color_scale):
    """
    Generate a comprehensive summary report of the analysis.

    Parameters:
    -----------
    processing_results : list
        Results from parallel processing
    base_folders : list
        List of base folder paths
    custom_labels : list
        Custom labels for comparison
    global_color_scale : tuple
        Global color scale used
    """
    print(f"\n=== GENERATING SUMMARY REPORT ===")

    # Create report directory
    report_dir = 'Analysis_Summary_Report'
    os.makedirs(report_dir, exist_ok=True)

    # Generate detailed report
    report_path = os.path.join(report_dir, 'Batch_Analysis_Summary.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PARALLEL BATCH ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CPU Cores Used: {cpu_count()}\n")
        f.write(f"Total Base Folders: {len(base_folders)}\n")
        f.write(f"Total Simulation Runs: {len(processing_results)}\n\n")

        f.write("COLOR SCALING CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Global Minimum: {global_color_scale[0]:.6e}\n")
        f.write(f"Global Maximum: {global_color_scale[1]:.6e}\n")
        f.write("Consistent color scaling applied to all heatmaps\n\n")

        f.write("BASE FOLDERS PROCESSED:\n")
        f.write("-" * 40 + "\n")
        for i, (folder, label) in enumerate(zip(base_folders, custom_labels)):
            f.write(f"{i + 1}. {label}: {folder}\n")
        f.write("\n")

        f.write("PROCESSING RESULTS:\n")
        f.write("-" * 40 + "\n")
        successful = sum(1 for r in processing_results if r['success'])
        failed = len(processing_results) - successful

        f.write(f"Successful runs: {successful}\n")
        f.write(f"Failed runs: {failed}\n")
        f.write(f"Success rate: {successful / len(processing_results) * 100:.1f}%\n\n")

        if failed > 0:
            f.write("FAILED RUNS DETAILS:\n")
            f.write("-" * 40 + "\n")
            for result in processing_results:
                if not result['success']:
                    f.write(f"- {result['label']}: {result.get('error', 'Unknown error')}\n")
            f.write("\n")

        f.write("SUCCESSFUL RUNS DETAILS:\n")
        f.write("-" * 40 + "\n")
        for result in processing_results:
            if result['success']:
                f.write(f"✓ {result['label']}\n")
                f.write(f"  - Standard plots: {result['standard_plots']}\n")
                f.write(f"  - Enhanced analysis: {result['enhanced_analysis']}\n")
                f.write(f"  - Process ID: {result['process_id']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"✅ Summary report saved to: {report_path}")




if __name__ == "__main__":
    # Ensure proper multiprocessing start method for Windows compatibility
    if __name__ == "__main__":
        mp.set_start_method('spawn', force=True)

    # Configuration
    base_folders = [
        # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T08_320dp_100d_dZ_19.415_dK_19.387',
        # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T05_variable speed_ speedK_1',
        # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run7_ReDimension_L57\T01_optimized_clearance'
        r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T05_variable speed_ speedK_1',
        # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run12_Optimal_value\5_para_dK_19.678_dZ_19.694_LKG_47_LF_32.5_zeta_4',
        # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run12_Optimal_value\3_para_dK_19.332_dZ_19.352_zeta_8',
        r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run12_Optimal_value\2_para',

    ]

    # Custom labels for comparison plots
    custom_labels = [
        "V32HL Industrial Design",
        # "V32HL 5-Parameter",
        # "V32HL 3-Parameter",
        "V32HL 2-Parameter",

    ]

    # Analysis options
    plot_types = 'all'  # or specific types: ['gap_height', 'contact_pressure', etc.]
    run_enhanced_analysis = True  # Set to False to skip enhanced contact pressure analysis
    create_comparisons = True  # Set to False to skip comparison plots
    clean_existing_plots = True  # Set to False to keep existing plots

    # Run the comprehensive batch analysis with maximum CPU utilization
    run_batch_analysis_parallel(
        base_folders=base_folders,
        plot_types=plot_types,
        custom_labels=custom_labels,
        run_enhanced_analysis=run_enhanced_analysis,
        create_comparisons=create_comparisons,
        clean_existing_plots=clean_existing_plots
    )