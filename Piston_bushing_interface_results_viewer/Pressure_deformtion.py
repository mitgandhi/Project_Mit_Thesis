# pressure_deformation.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.interpolate import interp1d

def load_matlab_txt(path):
    return np.genfromtxt(path, comments='%', delimiter=None)

def round2(x, base=0.6):
    return base * round(x / base)

def piston_pressure_deformation(filepath, m, lF, n, deltap, plots, degstep, ognore, minplot, maxplot, offset):
    ignore = ognore + offset
    data = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'matlab', 'Piston_Gap_Pressure_Deformation.txt'))
    gaplength = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'piston.txt'))

    xdata = np.linspace(0, 360, m)
    lastplot = degstep * len(data) / m - 1
    while lastplot > 360:
        lastplot -= 360

    lastline = len(data) - ignore * m
    firstline = lastline - m * plots + 1
    maxdata = np.max(data[firstline-1:lastline, :])
    mindata = np.min(data[firstline-1:lastline, :])

    if minplot == 0:
        minplot = round2(mindata)
    if maxplot == 0:
        maxplot = round2(maxdata)

    limit = len(gaplength)
    for i in range(len(gaplength) - 1):
        if gaplength[i, 2] > gaplength[i+1, 2]:
            limit = i
            break

    out_dir = os.path.join(filepath, 'output', 'piston', 'Plots', 'Piston_Pressure_Deformation')
    os.makedirs(out_dir, exist_ok=True)

    meanDef, maxDef, minDef = [], [], []
    globalMaxVal, globalMinVal = -np.inf, np.inf
    maxFrameData, minFrameData = None, None

    print(f"\nGenerating pressure deformation plots for: {filepath}")
    progress = 0

    for i in range(plots):
        deg = lastplot - degstep * (plots - i + ignore)
        while deg < 0:
            deg += 360

        interp_len = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(deg)
        interp_off = interp1d(gaplength[:limit, 2], gaplength[:limit, 18], fill_value='extrapolate')(deg)
        ydata = np.linspace(0, interp_len, data.shape[1]) + interp_off

        frame = data[firstline - 1 + i * m : firstline - 1 + (i + 1) * m, :]
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
        plt.savefig(os.path.join(out_dir, f'{i+1}.jpg'))
        plt.close()

        meanDef.append(np.mean(frame))
        maxDef.append(current_max)
        minDef.append(current_min)

        if int(50 * (i+1) / plots) > progress:
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