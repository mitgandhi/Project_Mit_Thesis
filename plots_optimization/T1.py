import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat
import csv


def load_matlab_txt(path):
    return np.genfromtxt(path, comments='%', delimiter=None)


def round2(x, base=0.6):
    return base * round(x / base)


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


def create_cumulative_pressure_map(filepath, m, lF, n, deltap, plots, degstep, ignore, offset,
                                   minplot=0, maxplot=0):
    """
    Create cumulative pressure maps with consistent scaling as in piston_contact_pressure.py

    Parameters:
    - filepath: Path to the simulation files
    - m: Number of circumferential points
    - lF: Axial length of piston (mm), converted to meters in the code
    - n: RPM value for plot titles
    - deltap: Pressure difference for plot titles
    - plots: Number of plots/frames to process
    - degstep: Angular step between frames in degrees
    - ignore: Number of frames to ignore from the end
    - offset: Additional offset to apply
    - minplot: Minimum pressure limit (0 means auto-calculate using round2)
    - maxplot: Maximum pressure limit (0 means auto-calculate using round2)
    """
    # ===== STEP 1: DATA LOADING AND PREPARATION =====
    try:
        data = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt'))
        gaplength = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'piston.txt'))
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None

    # Create basic coordinate arrays
    xdata = np.linspace(0, 360, m)
    lastplot = degstep * len(data) / m - 1
    while lastplot > 360:
        lastplot -= 360

    # Apply offset to ignore
    ignore_adjusted = ignore + offset

    # Calculate data range for analysis
    lastline = len(data) - ignore_adjusted * m
    firstline = lastline - m * plots + 1

    # Validate data range
    if firstline < 0 or lastline > len(data):
        print(
            f"Invalid data range. Check parameters: firstline={firstline}, lastline={lastline}, data length={len(data)}")
        return None

    # Get data range for consistent scaling with code 1
    data_range = data[firstline - 1:lastline, :]
    maxdata = np.max(data_range)
    mindata = np.min(data_range)

    # Use the same rounding logic for plot limits
    if minplot == 0:
        minplot = round2(mindata)
    if maxplot == 0:
        maxplot = round2(maxdata)

    # Find gap length limit
    limit = len(gaplength)
    for i in range(len(gaplength) - 1):
        if gaplength[i, 2] > gaplength[i + 1, 2]:
            limit = i
            break

    # ===== STEP 2: DATA PROCESSING =====
    # Initialize pressure maps
    cumulative_pressure_map = np.zeros((m, int(lF * 10)))
    max_pressure_map = np.zeros((m, int(lF * 10)))

    # Setup coordinates for plotting
    y_coords = np.linspace(0, lF * 1e-3, int(lF * 10))
    x_coords = np.linspace(0, 360, m)
    degrees = np.linspace(0, 360, m)

    print("\nProcessing pressure data for cumulative and max heatmaps...")
    progress = 0

    # Process each frame
    for i in range(plots):
        deg = degstep * (i + ignore_adjusted) % 360

        interp_len = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(deg)
        interp_off = interp1d(gaplength[:limit, 2], gaplength[:limit, 18], fill_value='extrapolate')(deg)
        ydata = np.linspace(0, interp_len, data.shape[1]) + interp_off

        frame = data[firstline - 1 + i * m: firstline - 1 + (i + 1) * m, :]

        for j in range(m):
            for k, y in enumerate(ydata):
                if y <= lF * 1e-3:
                    y_idx = min(int(y * 10000), int(lF * 10) - 1)
                    val = frame[j, k]
                    cumulative_pressure_map[j, y_idx] += val
                    max_pressure_map[j, y_idx] = max(max_pressure_map[j, y_idx], val)

        if int(50 * (i + 1) / plots) > progress:
            print('-', end='', flush=True)
            progress += 1

    # Calculate mean pressure map
    mean_pressure_map = np.divide(
        cumulative_pressure_map,
        plots,
        out=np.zeros_like(cumulative_pressure_map),
        where=cumulative_pressure_map != 0
    )

    # Calculate ratio (max / cumulative)
    ratio_map = np.divide(
        max_pressure_map,
        cumulative_pressure_map,
        out=np.zeros_like(max_pressure_map),
        where=cumulative_pressure_map != 0
    )

    # Calculate per-angle statistics
    meanPressPerAngle = np.mean(mean_pressure_map, axis=1)
    maxPressPerAngle = np.max(max_pressure_map, axis=1)
    cumulativePressPerAngle = np.sum(cumulative_pressure_map, axis=1)

    # Create summary dictionary
    summary = {
        'meanContactPressure': meanPressPerAngle,
        'maxContactPressure': maxPressPerAngle,
        'cumulativeContactPressure': cumulativePressPerAngle,
        'degrees': degrees,
        'maxValue': np.max(maxPressPerAngle),
        'minValue': np.min(maxPressPerAngle),
        'phiMax': degrees[np.argmax(maxPressPerAngle)],
        'phiMin': degrees[np.argmin(maxPressPerAngle)]
    }

    # Setup output directory
    out_dir = os.path.join(filepath, 'output', 'piston', 'Plots', 'Piston_Contact_Pressure_cumulative')
    os.makedirs(out_dir, exist_ok=True)

    # ===== STEP 3: SAVE MATLAB DATA =====
    savemat(os.path.join(out_dir, 'contact_pressure_data.mat'), {'comparisonData': summary})

    # ===== STEP 4: PRINT STATISTICS =====
    # Print summary statistics
    print(f"\nMax Contact Pressure: {np.max(maxPressPerAngle):.2f} Pa @ {degrees[np.argmax(maxPressPerAngle)]:.1f}°")
    print(f"Min Contact Pressure: {np.min(maxPressPerAngle):.2f} Pa @ {degrees[np.argmin(maxPressPerAngle)]:.1f}°")
    print(f"Mean Contact Pressure: {np.mean(mean_pressure_map):.2f} Pa")
    print(f"Total Cumulative Contact Pressure: {np.sum(cumulative_pressure_map):.2f} Pa")

    # ===== STEP 5: INDIVIDUAL PLOTS =====
    # Define plotting function
    def plot_and_save(data, title, filename, label, vmin=minplot, vmax=maxplot):
        plt.figure(figsize=(12, 8))
        c = plt.pcolormesh(y_coords, x_coords, data, shading='auto', cmap='jet')
        c.set_clim(vmin, vmax)  # Match code 1's approach to setting limits
        cbar = plt.colorbar(c, label=label)
        cbar.ax.tick_params(labelsize=16)
        plt.title(title)
        plt.xlabel('Gap Length [m]', fontsize=16)
        plt.ylabel('Gap Circumference [°]', fontsize=16)
        plt.yticks(np.arange(0, 361, 60))  # Match code 1's y-ticks
        plt.ylim([0, 360])  # Match code 1's y-axis limits
        plt.xlim([0, lF * 1e-3])  # Match code 1's x-axis limits
        plt.xticks(np.linspace(0, lF * 1e-3, 6))  # Match code 1's x-ticks approach
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()

    # Cumulative map
    plot_and_save(
        cumulative_pressure_map,
        f'Cumulative Piston Contact Pressure\nn={n}, ?P={deltap}',
        'Cumulative_Pressure.png',
        'Cumulative Pressure [Pa]'
    )

    # Mean map
    plot_and_save(
        mean_pressure_map,
        f'Mean Piston Contact Pressure\nn={n}, ?P={deltap}',
        'Mean_Pressure.png',
        'Mean Pressure [Pa]'
    )

    # Max map
    plot_and_save(
        max_pressure_map,
        f'Maximum Piston Contact Pressure\nn={n}, ?P={deltap}',
        'Max_Pressure.png',
        'Maximum Pressure [Pa]'
    )

    # Ratio map with contour factors
    plt.figure(figsize=(12, 8))
    c = plt.pcolormesh(y_coords, x_coords, ratio_map, shading='auto', cmap='jet')
    plt.colorbar(label='Max / Cumulative Pressure Ratio (Factor)')

    contour_levels = [0.5, 1, 2, 5, 10]
    CS = plt.contour(y_coords, x_coords, ratio_map, levels=contour_levels, colors='white', linewidths=0.75)
    plt.clabel(CS, inline=True, fontsize=8, fmt='%.1f')

    plt.title(f'Max-to-Cumulative Pressure Ratio (Factor)\nn={n}, ?P={deltap}')
    plt.xlabel('Gap Length [m]', fontsize=16)
    plt.ylabel('Gap Circumference [°]', fontsize=16)
    plt.yticks(np.arange(0, 361, 60))  # Match code 1's y-ticks
    plt.ylim([0, 360])  # Match code 1's y-axis limits
    plt.xlim([0, lF * 1e-3])  # Match code 1's x-axis limits
    plt.xticks(np.linspace(0, lF * 1e-3, 6))  # Match code 1's x-ticks approach
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Max_to_Cumulative_Ratio.png'), dpi=300)
    plt.close()

    # ===== STEP 6: 1D LINE PLOTS =====
    # Plot mean pressure vs shaft angle similar to code 1
    plt.figure()
    plt.plot(degrees, meanPressPerAngle, 'b-', linewidth=1.5)
    plt.title('Mean Contact Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Mean Contact Pressure [Pa]')
    plt.grid(True)
    plt.xticks(np.arange(0, 361, 60))
    plt.xlim(0, 360)
    plt.savefig(os.path.join(out_dir, 'Mean_Contact_Pressure_vs_Shaft_Angle.png'))
    plt.close()

    # Plot max pressure vs shaft angle similar to code 1
    plt.figure()
    plt.plot(degrees, maxPressPerAngle, 'r-', linewidth=1.5)
    plt.title('Max Contact Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Max Contact Pressure [Pa]')
    plt.grid(True)
    plt.xticks(np.arange(0, 361, 60))
    plt.xlim(0, 360)
    plt.savefig(os.path.join(out_dir, 'Max_Contact_Pressure_vs_Shaft_Angle.png'))
    plt.close()

    # Plot cumulative pressure vs shaft angle (same as code 1)
    plt.figure()
    plt.plot(degrees, cumulativePressPerAngle, 'g-', linewidth=1.5)
    plt.title('Cumulative Contact Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Cumulative Contact Pressure [Pa·m²]')
    plt.grid(True)
    plt.xticks(np.arange(0, 361, 60))
    plt.xlim(0, 360)
    plt.savefig(os.path.join(out_dir, 'Cumulative_Contact_Pressure_vs_Shaft_Angle.png'))
    plt.close()

    # ===== STEP 7: COMBINED SUBPLOT FIGURE =====
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)

    # First subplot: Cumulative Pressure
    c1 = axs[0, 0].pcolormesh(y_coords, x_coords, cumulative_pressure_map, shading='auto', cmap='jet')
    c1.set_clim(minplot, maxplot)  # Use consistent limits
    fig.colorbar(c1, ax=axs[0, 0], label='Cumulative Pressure [Pa]')
    axs[0, 0].set_title('Cumulative Pressure')
    axs[0, 0].set_xlabel('Gap Length [m]')
    axs[0, 0].set_ylabel('Gap Circumference [°]')
    axs[0, 0].set_yticks([0, 60, 120, 180, 240, 300, 360])
    axs[0, 0].set_ylim([0, 360])
    axs[0, 0].set_xlim([0, lF * 1e-3])
    axs[0, 0].set_xticks(np.linspace(0, lF * 1e-3, 6))
    axs[0, 0].axhline(180, color='black', linestyle='--', linewidth=1.5)

    # Second subplot: Mean Pressure
    c2 = axs[0, 1].pcolormesh(y_coords, x_coords, mean_pressure_map, shading='auto', cmap='jet')
    c2.set_clim(minplot, maxplot)  # Use consistent limits
    fig.colorbar(c2, ax=axs[0, 1], label='Mean Pressure [Pa]')
    axs[0, 1].set_title('Mean Pressure')
    axs[0, 1].set_xlabel('Gap Length [m]')
    axs[0, 1].set_ylabel('Gap Circumference [°]')
    axs[0, 1].set_yticks([0, 60, 120, 180, 240, 300, 360])
    axs[0, 1].set_ylim([0, 360])
    axs[0, 1].set_xlim([0, lF * 1e-3])
    axs[0, 1].set_xticks(np.linspace(0, lF * 1e-3, 6))

    # Third subplot: Max Pressure
    c3 = axs[1, 0].pcolormesh(y_coords, x_coords, max_pressure_map, shading='auto', cmap='jet')
    c3.set_clim(minplot, maxplot)  # Use consistent limits
    fig.colorbar(c3, ax=axs[1, 0], label='Max Pressure [Pa]')
    axs[1, 0].set_title('Maximum Pressure')
    axs[1, 0].set_xlabel('Gap Length [m]')
    axs[1, 0].set_ylabel('Gap Circumference [°]')
    axs[1, 0].set_yticks([0, 60, 120, 180, 240, 300, 360])
    axs[1, 0].set_ylim([0, 360])
    axs[1, 0].set_xlim([0, lF * 1e-3])
    axs[1, 0].set_xticks(np.linspace(0, lF * 1e-3, 6))

    # Fourth subplot: Ratio Map (different scale)
    c4 = axs[1, 1].pcolormesh(y_coords, x_coords, ratio_map, shading='auto', cmap='jet')
    fig.colorbar(c4, ax=axs[1, 1], label='Ratio (Factor)')
    axs[1, 1].set_title('Max/Cumulative Ratio (Factor)')
    axs[1, 1].set_xlabel('Gap Length [m]')
    axs[1, 1].set_ylabel('Gap Circumference [°]')
    axs[1, 1].set_yticks([0, 60, 120, 180, 240, 300, 360])
    axs[1, 1].set_ylim([0, 360])
    axs[1, 1].set_xlim([0, lF * 1e-3])
    axs[1, 1].set_xticks(np.linspace(0, lF * 1e-3, 6))

    plt.savefig(os.path.join(out_dir, 'Combined_Heatmaps.png'), dpi=300)
    plt.close()

    # ===== STEP 8: 3D TOP-DOWN VIEW =====
    print("Rendering 3D top-down view...")
    try:
        theta = np.linspace(0, 2 * np.pi, m)
        z = np.linspace(0, lF * 1e-3, cumulative_pressure_map.shape[1])
        theta_grid, z_grid = np.meshgrid(theta, z, indexing='ij')

        R = 1
        X = R * np.cos(theta_grid)
        Y = R * np.sin(theta_grid)
        Z = z_grid

        # Normalize data for coloring
        norm_data = np.copy(cumulative_pressure_map)
        norm_data = np.clip(norm_data, minplot, maxplot)  # Match code 1's approach
        norm_data = (norm_data - minplot) / (maxplot - minplot) if (maxplot - minplot) > 0 else 0.5

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(norm_data), rstride=1, cstride=1, antialiased=False, shade=False)

        offset = 0.05
        ax.plot([R + offset] * z.shape[0], [0] * z.shape[0], z, 'red', linewidth=3, label='0° Marker')
        ax.text(R + 0.1, 0, z[-1], '0°', color='red', fontsize=12)

        ax.view_init(elev=90, azim=-90)
        ax.set_title(f"3D Pressure on Cylinder (XY View)\nn={n}, ?P={deltap}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '3D_Top_Down_View.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error rendering 3D view: {e}")

    print("\n? All plots successfully generated!")

    return {
        'cumulative': cumulative_pressure_map,
        'mean': mean_pressure_map,
        'max': max_pressure_map,
        'ratio': ratio_map,
        'minplot': minplot,
        'maxplot': maxplot,
        'meanPressPerAngle': meanPressPerAngle,
        'maxPressPerAngle': maxPressPerAngle,
        'cumulativePressPerAngle': cumulativePressPerAngle,
        'summary': summary
    }


def collect_simulation_data(base_folders):
    """
    Collect all simulation runs from base folders and extract their parameters.
    Handles nested structure with 'optimized' and 'industrial_design' subfolders.

    Parameters:
    -----------
    base_folders : list
        List of base folder paths to scan for simulation runs

    Returns:
    --------
    list : List of dictionaries containing run data
    """
    all_runs_data = []

    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"Warning: {base_folder} not found; skipping.")
            continue

        # Check if this folder has the nested structure (optimized/industrial_design)
        optimized_path = os.path.join(base_folder, 'optimal')
        industrial_path = os.path.join(base_folder, 'industrial')

        if os.path.isdir(optimized_path) and os.path.isdir(industrial_path):
            print(f"Found nested structure in {base_folder}")

            # Get simulation folders from both subfolders
            optimized_sims = [d for d in os.listdir(optimized_path)
                              if os.path.isdir(os.path.join(optimized_path, d))]
            industrial_sims = [d for d in os.listdir(industrial_path)
                               if os.path.isdir(os.path.join(industrial_path, d))]

            # Process optimized simulations
            for sim_name in optimized_sims:
                fp = os.path.join(optimized_path, sim_name)
                geom_file = os.path.join(fp, 'input', 'geometry.txt')
                op_file = os.path.join(fp, 'input', 'operatingconditions.txt')

                lF_val = parse_geometry(geom_file)
                speed_val, hp_val = parse_operating_conditions(op_file)

                if lF_val is not None and speed_val is not None and hp_val is not None:
                    run_data = {
                        'filepath': fp,
                        'label': f"optimized_{sim_name}",
                        'sim_name': sim_name,
                        'type': 'optimized',
                        'lF': lF_val,
                        'n': speed_val,
                        'deltap': hp_val,
                        'm': 50,
                        'offset': 0,
                        'base_folder': base_folder
                    }
                    all_runs_data.append(run_data)
                else:
                    print(f"Warning: Skipping optimized '{sim_name}' (missing geometry or operating conditions)")

            # Process industrial design simulations
            for sim_name in industrial_sims:
                fp = os.path.join(industrial_path, sim_name)
                geom_file = os.path.join(fp, 'input', 'geometry.txt')
                op_file = os.path.join(fp, 'input', 'operatingconditions.txt')

                lF_val = parse_geometry(geom_file)
                speed_val, hp_val = parse_operating_conditions(op_file)

                if lF_val is not None and speed_val is not None and hp_val is not None:
                    run_data = {
                        'filepath': fp,
                        'label': f"industrial_{sim_name}",
                        'sim_name': sim_name,
                        'type': 'industrial_design',
                        'lF': lF_val,
                        'n': speed_val,
                        'deltap': hp_val,
                        'm': 50,
                        'offset': 0,
                        'base_folder': base_folder
                    }
                    all_runs_data.append(run_data)
                else:
                    print(f"Warning: Skipping industrial '{sim_name}' (missing geometry or operating conditions)")

        else:
            # Original flat structure - process directly
            subdirs = [d for d in os.listdir(base_folder)
                       if os.path.isdir(os.path.join(base_folder, d))]

            if not subdirs:
                print(f"Warning: No runs in {base_folder}; skipping.")
                continue

            for subdir in subdirs:
                fp = os.path.join(base_folder, subdir)
                geom_file = os.path.join(fp, 'input', 'geometry.txt')
                op_file = os.path.join(fp, 'input', 'operatingconditions.txt')

                lF_val = parse_geometry(geom_file)
                speed_val, hp_val = parse_operating_conditions(op_file)

                if lF_val is not None and speed_val is not None and hp_val is not None:
                    run_data = {
                        'filepath': fp,
                        'label': subdir,
                        'sim_name': subdir,
                        'type': 'standard',
                        'lF': lF_val,
                        'n': speed_val,
                        'deltap': hp_val,
                        'm': 50,
                        'offset': 0,
                        'base_folder': base_folder
                    }
                    all_runs_data.append(run_data)
                else:
                    print(f"Warning: Skipping '{subdir}' (missing geometry or operating conditions)")

    return all_runs_data


def create_comparison_plots(results, base_folder):
    """
    Create comparison plots between optimized and industrial design simulations.

    Parameters:
    -----------
    results : list
        List of processing results
    base_folder : str
        Base folder path where comparison plots will be saved
    """
    print(f"\nCreating comparison plots for {base_folder}")

    # Create comparison output directory
    comparison_dir = os.path.join(base_folder, 'Comparison_Optimized_vs_Industrial')
    os.makedirs(comparison_dir, exist_ok=True)

    # Group results by simulation name and type
    optimized_results = {}
    industrial_results = {}

    for result in results:
        if result['success'] and 'sim_name' in result:
            sim_name = result['sim_name']
            if result['type'] == 'optimized':
                optimized_results[sim_name] = result
            elif result['type'] == 'industrial_design':
                industrial_results[sim_name] = result

    # Find matching simulation names
    matching_sims = set(optimized_results.keys()) & set(industrial_results.keys())

    if not matching_sims:
        print("No matching simulations found between optimized and industrial designs")
        return

    print(f"Found {len(matching_sims)} matching simulations for comparison")

    # Create comparison plots for each matching simulation
    for sim_name in matching_sims:
        opt_result = optimized_results[sim_name]['result']
        ind_result = industrial_results[sim_name]['result']

        # Create individual comparison directory
        sim_comparison_dir = os.path.join(comparison_dir, f'Comparison_{sim_name}')
        os.makedirs(sim_comparison_dir, exist_ok=True)

        # Get the data and ensure both have the same dimensions
        opt_cumulative = opt_result['cumulative']
        ind_cumulative = ind_result['cumulative']

        # Debug: Print shapes to understand the issue
        print(f"Optimized cumulative shape: {opt_cumulative.shape}")
        print(f"Industrial cumulative shape: {ind_cumulative.shape}")

        # Ensure both arrays have the same shape - use the smaller dimensions
        min_rows = min(opt_cumulative.shape[0], ind_cumulative.shape[0])
        min_cols = min(opt_cumulative.shape[1], ind_cumulative.shape[1])

        opt_cumulative = opt_cumulative[:min_rows, :min_cols]
        ind_cumulative = ind_cumulative[:min_rows, :min_cols]

        print(f"Adjusted shapes - Opt: {opt_cumulative.shape}, Ind: {ind_cumulative.shape}")

        # Create proper coordinates that match the data dimensions
        # For pcolormesh with shading='auto', coordinates should match data dimensions exactly
        m, n_axial = opt_cumulative.shape

        # Create coordinate arrays
        y_coords = np.linspace(0, n_axial / 10000, n_axial)  # Gap length coordinates
        x_coords = np.linspace(0, 360, m)  # Circumference coordinates (degrees)

        # 1. Side-by-side cumulative pressure comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Optimized plot
        c1 = ax1.pcolormesh(y_coords, x_coords, opt_cumulative, shading='auto', cmap='jet')
        c1.set_clim(opt_result['minplot'], opt_result['maxplot'])
        ax1.set_title(f'Optimized - Cumulative Pressure\n{sim_name}')
        ax1.set_xlabel('Gap Length [m]')
        ax1.set_ylabel('Gap Circumference [°]')
        ax1.set_yticks([0, 60, 120, 180, 240, 300, 360])
        ax1.set_ylim([0, 360])
        plt.colorbar(c1, ax=ax1, label='Cumulative Pressure [Pa]')

        # Industrial plot
        c2 = ax2.pcolormesh(y_coords, x_coords, ind_cumulative, shading='auto', cmap='jet')
        c2.set_clim(ind_result['minplot'], ind_result['maxplot'])
        ax2.set_title(f'Industrial Design - Cumulative Pressure\n{sim_name}')
        ax2.set_xlabel('Gap Length [m]')
        ax2.set_ylabel('Gap Circumference [°]')
        ax2.set_yticks([0, 60, 120, 180, 240, 300, 360])
        ax2.set_ylim([0, 360])
        plt.colorbar(c2, ax=ax2, label='Cumulative Pressure [Pa]')

        plt.tight_layout()
        plt.savefig(os.path.join(sim_comparison_dir, f'{sim_name}_Cumulative_Pressure_Comparison.png'), dpi=300)
        plt.close()

        # 2. Difference plot (Optimized - Industrial)
        pressure_diff = opt_cumulative - ind_cumulative

        plt.figure(figsize=(12, 8))
        c = plt.pcolormesh(y_coords, x_coords, pressure_diff, shading='auto', cmap='RdBu_r')
        plt.colorbar(c, label='Pressure Difference [Pa]\n(Optimized - Industrial)')
        plt.title(f'Pressure Difference: Optimized - Industrial\n{sim_name}')
        plt.xlabel('Gap Length [m]')
        plt.ylabel('Gap Circumference [°]')
        plt.yticks([0, 60, 120, 180, 240, 300, 360])
        plt.ylim([0, 360])
        plt.tight_layout()
        plt.savefig(os.path.join(sim_comparison_dir, f'{sim_name}_Pressure_Difference.png'), dpi=300)
        plt.close()

        # 3. Line plot comparison of pressure vs shaft angle
        degrees = opt_result['summary']['degrees']

        plt.figure(figsize=(12, 6))
        plt.plot(degrees, opt_result['meanPressPerAngle'], 'b-', linewidth=2, label='Optimized - Mean')
        plt.plot(degrees, ind_result['meanPressPerAngle'], 'r-', linewidth=2, label='Industrial - Mean')
        plt.plot(degrees, opt_result['maxPressPerAngle'], 'b--', linewidth=1.5, label='Optimized - Max')
        plt.plot(degrees, ind_result['maxPressPerAngle'], 'r--', linewidth=1.5, label='Industrial - Max')

        plt.title(f'Contact Pressure vs. Shaft Angle Comparison\n{sim_name}')
        plt.xlabel('Shaft Angle [°]')
        plt.ylabel('Contact Pressure [Pa]')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(np.arange(0, 361, 60))
        plt.xlim(0, 360)
        plt.tight_layout()
        plt.savefig(os.path.join(sim_comparison_dir, f'{sim_name}_Pressure_vs_Angle_Comparison.png'), dpi=300)
        plt.close()

        # 4. Statistical comparison table
        comparison_stats = {
            'Metric': [
                'Mean Cumulative Pressure [Pa]',
                'Max Cumulative Pressure [Pa]',
                'Min Cumulative Pressure [Pa]',
                'Mean Contact Pressure [Pa]',
                'Max Contact Pressure [Pa]',
                'Total Cumulative Pressure [Pa]'
            ],
            'Optimized': [
                np.mean(opt_cumulative),
                np.max(opt_cumulative),
                np.min(opt_cumulative),
                np.mean(opt_result['meanPressPerAngle']),
                np.max(opt_result['maxPressPerAngle']),
                np.sum(opt_cumulative)
            ],
            'Industrial': [
                np.mean(ind_cumulative),
                np.max(ind_cumulative),
                np.min(ind_cumulative),
                np.mean(ind_result['meanPressPerAngle']),
                np.max(ind_result['maxPressPerAngle']),
                np.sum(ind_cumulative)
            ]
        }

        # Calculate differences and percentage changes
        comparison_stats['Difference (Opt-Ind)'] = [
            comparison_stats['Optimized'][i] - comparison_stats['Industrial'][i]
            for i in range(len(comparison_stats['Optimized']))
        ]

        comparison_stats['% Change'] = [
            ((comparison_stats['Optimized'][i] - comparison_stats['Industrial'][i]) /
             comparison_stats['Industrial'][i] * 100) if comparison_stats['Industrial'][i] != 0 else 0
            for i in range(len(comparison_stats['Optimized']))
        ]

        # Save comparison statistics to CSV
        stats_file = os.path.join(sim_comparison_dir, f'{sim_name}_Comparison_Statistics.csv')
        with open(stats_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Optimized', 'Industrial', 'Difference (Opt-Ind)', '% Change'])
            for i in range(len(comparison_stats['Metric'])):
                writer.writerow([
                    comparison_stats['Metric'][i],
                    f"{comparison_stats['Optimized'][i]:.2e}",
                    f"{comparison_stats['Industrial'][i]:.2e}",
                    f"{comparison_stats['Difference (Opt-Ind)'][i]:.2e}",
                    f"{comparison_stats['% Change'][i]:.2f}%"
                ])

        print(f"  ? Created comparison for {sim_name}")

    # Create overall summary comparison
    create_overall_summary(matching_sims, optimized_results, industrial_results, comparison_dir)

    print(f"? All comparison plots saved to {comparison_dir}")


def create_overall_summary(matching_sims, optimized_results, industrial_results, comparison_dir):
    """Create overall summary comparing all simulations."""

    # Collect summary data
    sim_names = list(matching_sims)
    opt_mean_pressures = []
    ind_mean_pressures = []
    opt_max_pressures = []
    ind_max_pressures = []

    for sim_name in sim_names:
        opt_result = optimized_results[sim_name]['result']
        ind_result = industrial_results[sim_name]['result']

        opt_mean_pressures.append(np.mean(opt_result['meanPressPerAngle']))
        ind_mean_pressures.append(np.mean(ind_result['meanPressPerAngle']))
        opt_max_pressures.append(np.max(opt_result['maxPressPerAngle']))
        ind_max_pressures.append(np.max(ind_result['maxPressPerAngle']))

    # Create summary bar plots
    x = np.arange(len(sim_names))
    width = 0.35

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 6))

    # Mean pressure comparison
    bars1 = ax1.bar(x - width / 2, opt_mean_pressures, width, label='Optimized', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, ind_mean_pressures, width, label='Industrial', alpha=0.8)

    ax1.set_title('Mean Contact Pressure Comparison')
    ax1.set_xlabel('Simulation')
    ax1.set_ylabel('Mean Contact Pressure [Pa]')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sim_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Max pressure comparison
    bars3 = ax2.bar(x - width / 2, opt_max_pressures, width, label='Optimized', alpha=0.8)
    bars4 = ax2.bar(x + width / 2, ind_max_pressures, width, label='Industrial', alpha=0.8)

    ax2.set_title('Max Contact Pressure Comparison')
    ax2.set_xlabel('Simulation')
    ax2.set_ylabel('Max Contact Pressure [Pa]')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sim_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'Overall_Summary_Comparison.png'), dpi=300)
    plt.close()

    print("  ? Created overall summary comparison")


def run_contact_pressure_analysis(base_folders, plots=360, degstep=1, ignore=0, minplot=0, maxplot=0):
    """
    Main function to run contact pressure analysis on multiple base folders.
    Now includes comparison functionality for optimized vs industrial design.

    Parameters:
    -----------
    base_folders : list
        List of base folder paths containing simulation data
    plots : int, optional
        Number of plots/frames to process (default: 360)
    degstep : int, optional
        Angular step between frames in degrees (default: 1)
    ignore : int, optional
        Number of frames to ignore from the end (default: 0)
    minplot : float, optional
        Minimum pressure limit (0 for auto-calculate)
    maxplot : float, optional
        Maximum pressure limit (0 for auto-calculate)
    """

    print("=" * 60)
    print("CONTACT PRESSURE CUMULATIVE ANALYSIS")
    print("=" * 60)

    # Collect simulation data
    print("Collecting simulation data...")
    all_runs_data = collect_simulation_data(base_folders)

    if not all_runs_data:
        print("No valid simulation runs found. Exiting.")
        return

    print(f"Found {len(all_runs_data)} simulation runs to process")

    # Process each run
    results = []
    successful_runs = 0
    failed_runs = 0

    for i, run_data in enumerate(all_runs_data):
        print(f"\n[{i + 1}/{len(all_runs_data)}] Processing: {run_data['label']}")

        try:
            result = create_cumulative_pressure_map(
                filepath=run_data['filepath'],
                m=run_data['m'],
                lF=run_data['lF'],
                n=run_data['n'],
                deltap=run_data['deltap'],
                plots=plots,
                degstep=degstep,
                ignore=ignore,
                offset=run_data['offset'],
                minplot=minplot,
                maxplot=maxplot
            )

            if result is not None:
                results.append({
                    'label': run_data['label'],
                    'filepath': run_data['filepath'],
                    'sim_name': run_data.get('sim_name', run_data['label']),
                    'type': run_data.get('type', 'standard'),
                    'base_folder': run_data['base_folder'],
                    'result': result,
                    'success': True
                })
                successful_runs += 1
                print(f"? Successfully processed: {run_data['label']}")
            else:
                results.append({
                    'label': run_data['label'],
                    'filepath': run_data['filepath'],
                    'sim_name': run_data.get('sim_name', run_data['label']),
                    'type': run_data.get('type', 'standard'),
                    'base_folder': run_data['base_folder'],
                    'error': 'Processing returned None',
                    'success': False
                })
                failed_runs += 1
                print(f"? Failed to process: {run_data['label']}")

        except Exception as e:
            results.append({
                'label': run_data['label'],
                'filepath': run_data['filepath'],
                'sim_name': run_data.get('sim_name', run_data['label']),
                'type': run_data.get('type', 'standard'),
                'base_folder': run_data['base_folder'],
                'error': str(e),
                'success': False
            })
            failed_runs += 1
            print(f"? Error processing {run_data['label']}: {e}")

    # Group results by base folder for comparison
    results_by_base = {}
    for result in results:
        base_folder = result['base_folder']
        if base_folder not in results_by_base:
            results_by_base[base_folder] = []
        results_by_base[base_folder].append(result)

    # Create comparison plots for folders with optimized/industrial structure
    for base_folder, folder_results in results_by_base.items():
        # Check if this folder has both optimized and industrial results
        has_optimized = any(r.get('type') == 'optimized' for r in folder_results)
        has_industrial = any(r.get('type') == 'industrial_design' for r in folder_results)

        if has_optimized and has_industrial:
            print(f"\n" + "=" * 50)
            print(f"CREATING COMPARISONS FOR: {os.path.basename(base_folder)}")
            print("=" * 50)
            create_comparison_plots(folder_results, base_folder)

    # Print final summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total runs processed: {len(all_runs_data)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs / len(all_runs_data) * 100:.1f}%")

    if failed_runs > 0:
        print("\nFailed runs:")
        for result in results:
            if not result['success']:
                print(f"  - {result['label']}: {result.get('error', 'Unknown error')}")

    return results


if __name__ == "__main__":
    # Configuration - modify these paths to your simulation folders
    base_folders = [
        # r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\T05_variable speed_ speedK_1',
        r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run12_Optimal_value\5_para_t',
    ]

    # Analysis parameters
    plots = 360  # Number of frames to process
    degstep = 1  # Angular step in degrees
    ignore = 0  # Frames to ignore from end
    minplot = 0  # Minimum pressure limit (0 for auto)
    maxplot = 0  # Maximum pressure limit (0 for auto)

    # Run the analysis
    results = run_contact_pressure_analysis(
        base_folders=base_folders,
        plots=plots,
        degstep=degstep,
        ignore=ignore,
        minplot=minplot,
        maxplot=maxplot
    )

    print("\n? Contact pressure cumulative analysis completed!")