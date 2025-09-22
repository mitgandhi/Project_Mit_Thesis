# gap_height.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.interpolate import interp1d

def load_matlab_txt(path):
    return np.genfromtxt(path, comments='%', delimiter=None)

def piston_gap_height(filepath, m, lF, n, deltap, plots, degstep, ignore_base, minplot, maxplot, offset):
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
    print(f"\nGenerating gap height plots for: {filepath}")
    progress = 0

    for i in range(plots):
        deg = lastplot - degstep * (plots - i + ignore)
        while deg < 0:
            deg += 360

        interp_len = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(deg)
        interp_off = interp1d(gaplength[:limit, 2], gaplength[:limit, 18], fill_value='extrapolate')(deg)
        ydata = np.linspace(0, interp_len, data.shape[1]) + interp_off

        frame = data[firstline + i * m : firstline + (i + 1) * m, :]

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
        plt.savefig(os.path.join(out_dir, f'{i+1}.jpg'))
        plt.close()

        meanGap.append(np.mean(frame))
        maxGap.append(np.max(frame))
        minGap.append(np.min(frame))

        if int(50 * (i+1) / plots) > progress:
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

