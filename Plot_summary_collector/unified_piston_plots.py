# unified_piston_plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.interpolate import interp1d


def load_matlab_txt(path):
    """Load data from MATLAB text file format."""
    return np.genfromtxt(path, comments='%', delimiter=None)


def round2(x, base=0.6):
    """Round to nearest base value."""
    return base * round(x / base)


def generate_piston_plots(
        filepath,
        plot_type=None,  # If None, will detect from file presence
        m=50,
        lF=None,  # Length of gap
        n=None,  # Speed
        deltap=None,  # Pressure differential
        plots=360,
        degstep=1,
        ignore_base=0,
        minplot=0,
        maxplot=0,
        offset=0
):
    """
    Unified function to generate various piston plots based on the plot_type.

    Parameters:
    -----------
    filepath : str
        Path to the root folder containing the data
    plot_type : str, optional
        Type of plot to generate. If None, will detect from file presence.
        Options: 'gap_height', 'pressure_deformation', 'contact_pressure', 'all'
    m : int
        Number of points in circumferential direction
    lF : float
        Length of gap in mm
    n : float
        Speed in rpm
    deltap : float
        Pressure differential
    plots : int
        Number of plots to generate
    degstep : float
        Angle step in degrees
    ignore_base : int
        Number of frames to ignore from the end
    minplot, maxplot : float
        Min and max values for colorbar (0 for auto)
    offset : int
        Additional offset for frame selection
    """
    # Define the file mapping and detect plot types if not specified
    file_mappings = {
        'gap_height': 'Piston_Gap_Height.txt',
        'pressure_deformation': 'Piston_Gap_Pressure_Deformation.txt',
        'contact_pressure': 'Piston_Contact_Pressure.txt',
        'gap_pressure': 'Piston_Gap_Pressure.txt'
    }

    # Determine which plot types to generate
    plot_types_to_run = []

    if plot_type == 'all':
        # Check which files exist and add corresponding plot types
        for k, v in file_mappings.items():
            file_path = os.path.join(filepath, 'output', 'piston', 'matlab', v)
            if os.path.exists(file_path):
                plot_types_to_run.append(k)
    elif plot_type is None:
        # Auto-detect which files exist
        for k, v in file_mappings.items():
            file_path = os.path.join(filepath, 'output', 'piston', 'matlab', v)
            if os.path.exists(file_path):
                plot_types_to_run.append(k)
    else:
        # Use the specified plot type if the file exists
        if plot_type in file_mappings:
            file_path = os.path.join(filepath, 'output', 'piston', 'matlab', file_mappings[plot_type])
            if os.path.exists(file_path):
                plot_types_to_run.append(plot_type)

    if not plot_types_to_run:
        print(f"No valid plot types found for {filepath}")
        return

    # Generate each type of plot
    results = {}
    for plot_type in plot_types_to_run:
        print(f"\nProcessing {plot_type} for {filepath}")
        if plot_type == 'gap_height':
            result = _process_gap_height(filepath, m, lF, n, deltap, plots, degstep, ignore_base, minplot, maxplot,
                                         offset)
        elif plot_type == 'pressure_deformation':
            result = _process_pressure_deformation(filepath, m, lF, n, deltap, plots, degstep, ignore_base, minplot,
                                                   maxplot, offset)
        elif plot_type == 'contact_pressure':
            result = _process_contact_pressure(filepath, m, lF, n, deltap, plots, degstep, ignore_base, minplot,
                                               maxplot, offset)
        elif plot_type == 'gap_pressure':
            result = _process_gap_pressure(filepath, m, lF, n, deltap, plots, degstep, ignore_base, minplot, maxplot,
                                           offset)

        results[plot_type] = result

    return results


def _process_gap_height(filepath, m, lF, n, deltap, plots, degstep, ignore_base, minplot, maxplot, offset):
    """Process gap height plots (based on piston_gap_height.py)"""
    ignore = ignore_base + offset
    data = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'matlab', 'Piston_Gap_Height.txt'))
    gaplength = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'piston.txt'))

    xdata = np.linspace(0, 360, m)
    lastplot = degstep * len(data) / m - 1
    while lastplot > 360:
        lastplot -= 360

    lastline = len(data) - ignore * m
    firstline = lastline - m * plots
    maxdata = np.max(data[firstline:lastline, :])
    mindata = np.min(data[firstline:lastline, :])

    limit = len(gaplength)
    for i in range(len(gaplength) - 1):
        if gaplength[i, 2] > gaplength[i + 1, 2]:
            limit = i
            break

    out_dir = os.path.join(filepath, 'output', 'piston', 'Plots', 'Piston_Gap_Height')
    os.makedirs(out_dir, exist_ok=True)

    meanGap, maxGap, minGap = [], [], []
    print(f"Generating gap height plots for: {filepath}")
    progress = 0

    for i in range(plots):
        # With this:
        deg = degstep * i
        deg = deg % 360

        interp_len = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(deg)
        interp_off = interp1d(gaplength[:limit, 2], gaplength[:limit, 18], fill_value='extrapolate')(deg)
        ydata = np.linspace(0, interp_len, data.shape[1]) + interp_off

        frame = data[firstline + i * m: firstline + (i + 1) * m, :]

        f, ax = plt.subplots()
        c = ax.pcolormesh(ydata, xdata, frame, shading='auto', cmap='jet_r')
        c.set_clim(minplot or mindata, maxplot or maxdata)
        ax.set_title(f'Piston Gap Height\nn={n}, ΔP={deltap}, φ={deg:.1f}°')
        ax.set_xlabel('Gap Length [m]')
        ax.set_ylabel('Gap Circumference [degrees]')
        plt.colorbar(c, ax=ax, label='Gap Height [μm]')
        ax.set_xlim([0, lF * 1e-3])
        ax.set_ylim([0, 360])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{i + 1:03d}_phi_{deg:.1f}.jpg'))
        plt.close()

        meanGap.append(np.mean(frame))
        maxGap.append(np.max(frame))
        minGap.append(np.min(frame))

        if int(50 * (i + 1) / plots) > progress:
            print('-', end='', flush=True)
            progress += 1

    degrees = np.linspace(0, 360, plots)
    idxMax = np.argmax(maxGap)
    idxMin = np.argmin(minGap)

    summary = {
        'meanGap': np.array(meanGap),
        'maxGap': np.array(maxGap),
        'minGap': np.array(minGap),
        'degrees': degrees,
        'maxValue': maxGap[idxMax],
        'minValue': minGap[idxMin],
        'phiMax': degrees[idxMax],
        'phiMin': degrees[idxMin]
    }

    savemat(os.path.join(out_dir, 'gap_height_data.mat'), {'gapSummary': summary})

    # Plot summary curves
    plt.figure()
    plt.plot(degrees, summary['meanGap'], 'k-', linewidth=1.5)
    plt.title('Mean Gap Height vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Mean Gap Height [μm]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Mean_Gap_vs_Shaft_Angle.png'))
    plt.close()

    plt.figure()
    plt.plot(degrees, summary['maxGap'], 'r-', linewidth=1.5)
    plt.title('Max Gap Height vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Max Gap Height [μm]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Max_Gap_vs_Shaft_Angle.png'))
    plt.close()

    plt.figure()
    plt.plot(degrees, summary['minGap'], 'b-', linewidth=1.5)
    plt.title('Min Gap Height vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Min Gap Height [μm]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Min_Gap_vs_Shaft_Angle.png'))
    plt.close()

    print(f"\nMax Gap Height: {summary['maxValue']:.2f} µm @ {summary['phiMax']:.1f}°")
    print(f"Min Gap Height: {summary['minValue']:.2f} µm @ {summary['phiMin']:.1f}°")

    return summary


def _process_pressure_deformation(filepath, m, lF, n, deltap, plots, degstep, ignore_base, minplot, maxplot, offset):
    """Process pressure deformation plots (based on pressure_deformation.py)"""
    ignore = ignore_base + offset
    data = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'matlab', 'Piston_Gap_Pressure_Deformation.txt'))
    gaplength = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'piston.txt'))

    xdata = np.linspace(0, 360, m)
    lastplot = degstep * len(data) / m - 1
    while lastplot > 360:
        lastplot -= 360

    lastline = len(data) - ignore * m
    firstline = lastline - m * plots + 1
    maxdata = np.max(data[firstline - 1:lastline, :])
    mindata = np.min(data[firstline - 1:lastline, :])

    if minplot == 0:
        minplot = round2(mindata)
    if maxplot == 0:
        maxplot = round2(maxdata)

    limit = len(gaplength)
    for i in range(len(gaplength) - 1):
        if gaplength[i, 2] > gaplength[i + 1, 2]:
            limit = i
            break

    out_dir = os.path.join(filepath, 'output', 'piston', 'Plots', 'Piston_Pressure_Deformation')
    os.makedirs(out_dir, exist_ok=True)

    meanDef, maxDef, minDef = [], [], []
    globalMaxVal, globalMinVal = -np.inf, np.inf
    maxFrameData, minFrameData = None, None
    phiMax = phiMin = 0

    print(f"Generating pressure deformation plots for: {filepath}")
    progress = 0

    for i in range(plots):
        deg = degstep * i
        deg = deg % 360

        interp_len = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(deg)
        interp_off = interp1d(gaplength[:limit, 2], gaplength[:limit, 18], fill_value='extrapolate')(deg)
        ydata = np.linspace(0, interp_len, data.shape[1]) + interp_off

        frame = data[firstline - 1 + i * m: firstline - 1 + (i + 1) * m, :]
        current_max = np.max(frame)
        current_min = np.min(frame)

        if current_max > globalMaxVal:
            globalMaxVal = current_max
            maxFrameData = frame
            phiMax = deg
        if current_min < globalMinVal:
            globalMinVal = current_min
            minFrameData = frame
            phiMin = deg

        f, ax = plt.subplots()
        c = ax.pcolormesh(ydata, xdata, frame, shading='auto', cmap='jet')
        c.set_clim(minplot, maxplot)
        ax.set_title(f'Piston Pressure Deformation\nn={n}, ΔP={deltap}, φ={deg:.1f}°')
        ax.set_xlabel('Gap Length [m]')
        ax.set_ylabel('Gap Circumference [degrees]')
        plt.colorbar(c, ax=ax, label='Pressure Deformation [μm]')
        ax.set_xlim([0, lF * 1e-3])
        ax.set_ylim([0, 360])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{i + 1:03d}_phi_{deg:.1f}.jpg'))
        plt.close()

        meanDef.append(np.mean(frame))
        maxDef.append(current_max)
        minDef.append(current_min)

        if int(50 * (i + 1) / plots) > progress:
            print('-', end='', flush=True)
            progress += 1

    degrees = np.linspace(0, 360, plots)
    summary = {
        'meanDeformation': np.array(meanDef),
        'maxDeformation': np.array(maxDef),
        'minDeformation': np.array(minDef),
        'degrees': degrees,
        'maxValue': globalMaxVal,
        'minValue': globalMinVal,
        'phiMax': phiMax,
        'phiMin': phiMin
    }

    savemat(os.path.join(out_dir, 'deformation_data.mat'), {'comparisonData': summary})

    # Plot summary curves
    plt.figure()
    plt.plot(degrees, summary['minDeformation'], 'b-', linewidth=1.5)
    plt.title('Min Gap Deformation vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Min Gap Deformation [μm]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Min_Gap_vs_Shaft_Angle.png'))
    plt.close()

    plt.figure()
    plt.plot(degrees, summary['maxDeformation'], 'r-', linewidth=1.5)
    plt.title('Max Gap Deformation vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Max Gap Deformation [μm]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Max_Gap_vs_Shaft_Angle.png'))
    plt.close()

    print(f"\nMax Deformation: {globalMaxVal:.2f} µm @ {phiMax:.1f}°")
    print(f"Min Deformation: {globalMinVal:.2f} µm @ {phiMin:.1f}°")

    return summary


def _process_contact_pressure(filepath, m, lF, n, deltap, plots, degstep, ignore_base, minplot, maxplot, offset):
    """Process contact pressure plots (based on contact_pressure.py)"""
    ignore = ignore_base + offset
    data = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt'))
    gaplength = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'piston.txt'))

    xdata = np.linspace(0, 360, m)
    lastplot = degstep * len(data) / m - 1
    while lastplot > 360:
        lastplot -= 360

    lastline = len(data) - ignore * m
    firstline = lastline - m * plots + 1
    maxdata = np.max(data[firstline - 1:lastline, :])
    mindata = np.min(data[firstline - 1:lastline, :])

    if minplot == 0:
        minplot = round2(mindata)
    if maxplot == 0:
        maxplot = round2(maxdata)

    limit = len(gaplength)
    for i in range(len(gaplength) - 1):
        if gaplength[i, 2] > gaplength[i + 1, 2]:
            limit = i
            break

    out_dir = os.path.join(filepath, 'output', 'piston', 'Plots', 'Piston_Contact_Pressure')
    os.makedirs(out_dir, exist_ok=True)

    meanPress, maxPress, minPress = [], [], []
    globalMaxVal, globalMinVal = -np.inf, np.inf
    phiMax = phiMin = 0

    print(f"Generating contact pressure plots for: {filepath}")
    progress = 0

    for i in range(plots):
        deg = degstep * (i + ignore)
        deg = deg % 360

        interp_len = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(deg)
        interp_off = interp1d(gaplength[:limit, 2], gaplength[:limit, 18], fill_value='extrapolate')(deg)
        ydata = np.linspace(0, interp_len, data.shape[1]) + interp_off

        frame = data[firstline - 1 + i * m: firstline - 1 + (i + 1) * m, :]
        current_max = np.max(frame)
        current_min = np.min(frame)

        if current_max > globalMaxVal:
            globalMaxVal = current_max
            phiMax = deg
        if current_min < globalMinVal:
            globalMinVal = current_min
            phiMin = deg

        f, ax = plt.subplots()
        c = ax.pcolormesh(ydata, xdata, frame, shading='auto', cmap='jet')
        c.set_clim(minplot, maxplot)
        ax.set_title(f'Piston Contact Pressure\nn={n}, ΔP={deltap}, φ={deg:.1f}°')
        ax.set_xlabel('Gap Length [m]')
        ax.set_ylabel('Gap Circumference [degrees]')
        plt.colorbar(c, ax=ax, label='Contact Pressure [Pa]')
        ax.set_yticks(np.arange(0, 361, 60))
        ax.set_ylim([0, 360])
        ax.set_xlim([0, lF * 1e-3])
        ax.set_xticks(np.linspace(0, lF * 1e-3, 6))

        max_idx = np.unravel_index(np.argmax(frame), frame.shape)
        max_y_angle = xdata[max_idx[0]]
        ax.axhline(max_y_angle, linestyle='--', color='white', linewidth=1.2)
        ax.text(0.02 * lF * 1e-3, max_y_angle + 5, f'Max φ ≈ {max_y_angle:.1f}°', color='white', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{i + 1:03d}_phi_{deg:.1f}.jpg'))
        plt.close()

        meanPress.append(np.mean(frame))
        maxPress.append(current_max)
        minPress.append(current_min)

        if int(50 * (i + 1) / plots) > progress:
            print('-', end='', flush=True)
            progress += 1

    degrees = np.linspace(0, 360, plots)
    # --- Accumulate contact pressure frame by frame ---
    cumulative_frame = np.zeros_like(data[firstline - 1: firstline - 1 + m, :])
    for i in range(plots):
        current_frame = data[firstline - 1 + i * m: firstline - 1 + (i + 1) * m, :]
        cumulative_frame += current_frame

    summary = {
        'meanContactPressure': np.array(meanPress),
        'maxContactPressure': np.array(maxPress),
        'minContactPressure': np.array(minPress),
        'degrees': degrees,
        'maxValue': globalMaxVal,
        'minValue': globalMinVal,
        'phiMax': phiMax,
        'phiMin': phiMin,
        'averageCumulativeFramePressure': np.mean(cumulative_frame)
    }

    savemat(os.path.join(out_dir, 'contact_pressure_data.mat'), {'comparisonData': summary})

    # Summary plots
    plt.figure()
    plt.plot(degrees, summary['meanContactPressure'], 'b-', linewidth=1.5)
    plt.title('Mean Contact Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Mean Contact Pressure [Pa]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Mean_Contact_Pressure_vs_Shaft_Angle.png'))
    plt.close()

    plt.figure()
    plt.plot(degrees, summary['maxContactPressure'], 'r-', linewidth=1.5)
    plt.title('Max Contact Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Max Contact Pressure [Pa]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Max_Contact_Pressure_vs_Shaft_Angle.png'))
    plt.close()

    print(f"\nMax Contact Pressure: {globalMaxVal:.2f} Pa @ {phiMax:.1f}°")
    print(f"Min Contact Pressure: {globalMinVal:.2f} Pa @ {phiMin:.1f}°")

    cumulativePress = [np.sum(data[firstline - 1 + i * m: firstline - 1 + (i + 1) * m, :]) for i in range(plots)]
    summary['cumulativeContactPressure'] = np.array(cumulativePress)

    plt.figure()
    plt.plot(degrees, cumulativePress, 'g-', linewidth=1.5)
    plt.title('Cumulative Contact Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Cumulative Contact Pressure [Pa·m²]')
    plt.grid(True)
    plt.xticks(np.arange(0, 361, 60))
    plt.xlim(0, 360)
    plt.savefig(os.path.join(out_dir, 'Cumulative_Contact_Pressure_vs_Shaft_Angle.png'))
    plt.close()

    # --- Plot cumulative heatmap ---
    mid_deg = degstep * (plots // 2 + ignore)
    mid_deg = mid_deg % 360
    interp_len = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(mid_deg)
    interp_off = interp1d(gaplength[:limit, 2], gaplength[:limit, 18], fill_value='extrapolate')(mid_deg)
    ydata = np.linspace(0, interp_len, data.shape[1]) + interp_off

    f, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(ydata, xdata, cumulative_frame, shading='auto', cmap='hot')
    ax.set_title(f'Cumulative Contact Pressure\nn={n}, ΔP={deltap}')
    ax.set_xlabel('Gap Length [m]')
    ax.set_ylabel('Gap Circumference [degrees]')
    plt.colorbar(c, ax=ax, label='Cumulative Pressure [Pa]')
    ax.set_xlim([ydata[0], ydata[-1]])
    ax.set_ylim([0, 360])
    ax.set_xticks(np.linspace(ydata[0], ydata[-1], 6))
    ax.set_yticks(np.arange(0, 361, 60))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Cumulative_Contact_Pressure_Heatmap.png'), bbox_inches='tight')
    plt.close()

    return summary


def _process_gap_pressure(filepath, m, lF, n, deltap, plots, degstep, ignore_base, minplot, maxplot, offset):
    """Process gap pressure plots"""
    ignore = ignore_base + offset
    data = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'matlab', 'Piston_Gap_Pressure.txt'))
    gaplength = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'piston.txt'))

    xdata = np.linspace(0, 360, m)
    lastplot = degstep * len(data) / m - 1
    while lastplot > 360:
        lastplot -= 360

    lastline = len(data) - ignore * m
    firstline = lastline - m * plots + 1
    maxdata = np.max(data[firstline - 1:lastline, :])
    mindata = np.min(data[firstline - 1:lastline, :])

    if minplot == 0:
        minplot = round2(mindata)
    if maxplot == 0:
        maxplot = round2(maxdata)

    limit = len(gaplength)
    for i in range(len(gaplength) - 1):
        if gaplength[i, 2] > gaplength[i + 1, 2]:
            limit = i
            break

    out_dir = os.path.join(filepath, 'output', 'piston', 'Plots', 'Piston_Gap_Pressure')
    os.makedirs(out_dir, exist_ok=True)

    meanPressure, maxPressure, minPressure = [], [], []
    globalMaxVal, globalMinVal = -np.inf, np.inf
    phiMax = phiMin = 0

    print(f"Generating gap pressure plots for: {filepath}")
    progress = 0

    for i in range(plots):
        deg = degstep * (i + ignore)
        deg = deg % 360

        interp_len = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(deg)
        interp_off = interp1d(gaplength[:limit, 2], gaplength[:limit, 18], fill_value='extrapolate')(deg)
        ydata = np.linspace(0, interp_len, data.shape[1]) + interp_off

        frame = data[firstline - 1 + i * m: firstline - 1 + (i + 1) * m, :]
        current_max = np.max(frame)
        current_min = np.min(frame)

        if current_max > globalMaxVal:
            globalMaxVal = current_max
            phiMax = deg
        if current_min < globalMinVal:
            globalMinVal = current_min
            phiMin = deg

        f, ax = plt.subplots()
        c = ax.pcolormesh(ydata, xdata, frame, shading='auto', cmap='jet')
        c.set_clim(minplot, maxplot)
        ax.set_title(f'Piston Gap Pressure\nn={n}, ΔP={deltap}, φ={deg:.1f}°')
        ax.set_xlabel('Gap Length [m]')
        ax.set_ylabel('Gap Circumference [degrees]')
        plt.colorbar(c, ax=ax, label='Pressure [Pa]')
        ax.set_yticks(np.arange(0, 361, 60))
        ax.set_ylim([0, 360])
        ax.set_xlim([0, lF * 1e-3])
        ax.set_xticks(np.linspace(0, lF * 1e-3, 6))

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{i + 1:03d}_phi_{deg:.1f}.jpg'))
        plt.close()

        meanPressure.append(np.mean(frame))
        maxPressure.append(current_max)
        minPressure.append(current_min)

        if int(50 * (i + 1) / plots) > progress:
            print('-', end='', flush=True)
            progress += 1

    degrees = np.linspace(0, 360, plots)
    summary = {
        'meanPressure': np.array(meanPressure),
        'maxPressure': np.array(maxPressure),
        'minPressure': np.array(minPressure),
        'degrees': degrees,
        'maxValue': globalMaxVal,
        'minValue': globalMinVal,
        'phiMax': phiMax,
        'phiMin': phiMin
    }

    savemat(os.path.join(out_dir, 'gap_pressure_data.mat'), {'pressureData': summary})

    # Plot summary curves
    plt.figure()
    plt.plot(degrees, summary['meanPressure'], 'b-', linewidth=1.5)
    plt.title('Mean Gap Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Mean Gap Pressure [Pa]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Mean_Pressure_vs_Shaft_Angle.png'))
    plt.close()

    plt.figure()
    plt.plot(degrees, summary['maxPressure'], 'r-', linewidth=1.5)
    plt.title('Max Gap Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Max Gap Pressure [Pa]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Max_Pressure_vs_Shaft_Angle.png'))
    plt.close()

    plt.figure()
    plt.plot(degrees, summary['minPressure'], 'g-', linewidth=1.5)
    plt.title('Min Gap Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Min Gap Pressure [Pa]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Min_Pressure_vs_Shaft_Angle.png'))
    plt.close()

    print(f"\nMax Gap Pressure: {globalMaxVal:.2f} Pa @ {phiMax:.1f}°")
    print(f"Min Gap Pressure: {globalMinVal:.2f} Pa @ {phiMin:.1f}°")

    return summary