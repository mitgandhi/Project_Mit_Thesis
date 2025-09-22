import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.interpolate import interp1d

def load_matlab_txt(path):
    return np.genfromtxt(path, comments='%', delimiter=None)

def round2(x, base=0.6):
    return base * round(x / base)

def piston_contact_pressure(filepath, m, lF, n, deltap, plots, degstep, ognore, minplot, maxplot, offset):
    ignore = ognore + offset
    data = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt'))
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

    out_dir = os.path.join(filepath, 'output', 'piston', 'Plots', 'Piston_Contact_Pressure')
    os.makedirs(out_dir, exist_ok=True)

    meanPress, maxPress, minPress = [], [], []
    globalMaxVal, globalMinVal = -np.inf, np.inf
    phiMax = phiMin = 0

    print(f"\nGenerating contact pressure plots for: {filepath}")
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
        ax.set_xlim([0, lF * 1e-3])
        ax.set_ylim([0, 360])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{i+1}.jpg'))
        plt.close()

        meanPress.append(np.mean(frame))
        maxPress.append(current_max)
        minPress.append(current_min)

        if int(50 * (i+1) / plots) > progress:
            print('-', end='', flush=True)
            progress += 1

    degrees = np.linspace(0, 360, plots)
    summary = {
        'meanContactPressure': np.array(meanPress),
        'maxContactPressure': np.array(maxPress),
        'minContactPressure': np.array(minPress),
        'degrees': degrees,
        'maxValue': globalMaxVal,
        'minValue': globalMinVal,
        'phiMax': phiMax,
        'phiMin': phiMin
    }

    savemat(os.path.join(out_dir, 'contact_pressure_data.mat'), {'comparisonData': summary})

    plt.figure()
    plt.plot(degrees, summary['minContactPressure'], 'b-', linewidth=1.5)
    plt.title('Min Contact Pressure vs. Shaft Angle')
    plt.xlabel('Shaft Angle [°]')
    plt.ylabel('Min Contact Pressure [Pa]')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'Min_Contact_Pressure_vs_Shaft_Angle.png'))
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
