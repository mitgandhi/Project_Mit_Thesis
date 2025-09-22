import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat


def load_matlab_txt(path):
    return np.genfromtxt(path, comments='%', delimiter=None)


def round2(x, base=0.6):
    return base * round(x / base)


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
        plt.colorbar(label=label,fontsize=16)
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
        f'Cumulative Piston Contact Pressure\nn={n}, ΔP={deltap}',
        'Cumulative_Pressure.png',
        'Cumulative Pressure [Pa]'
    )

    # Mean map
    plot_and_save(
        mean_pressure_map,
        f'Mean Piston Contact Pressure\nn={n}, ΔP={deltap}',
        'Mean_Pressure.png',
        'Mean Pressure [Pa]'
    )

    # Max map
    plot_and_save(
        max_pressure_map,
        f'Maximum Piston Contact Pressure\nn={n}, ΔP={deltap}',
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

    plt.title(f'Max-to-Cumulative Pressure Ratio (Factor)\nn={n}, ΔP={deltap}')
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
        ax.set_title(f"3D Pressure on Cylinder (XY View)\nn={n}, ΔP={deltap}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '3D_Top_Down_View.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error rendering 3D view: {e}")

    print("\n✅ All plots successfully generated!")

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