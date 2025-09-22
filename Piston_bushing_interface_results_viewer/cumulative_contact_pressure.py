import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_matlab_txt(path):
    return np.genfromtxt(path, comments='%', delimiter=None)

def cumulative_contact_pressure(
    filepath,
    m=100,
    dK=20.7,      # mm
    steps=360,
    degstep=1.0,
    ignore=0,
    bins=1000
):
    print(f"\nGenerating cumulative contact pressure curve for: {filepath}")
    dK = dK * 0.001  # Convert to meters

    data = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt'))
    gaplength = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'piston.txt'))

    lastline = len(data) - ignore * m
    firstline = lastline - m * steps + 1

    lastplot = degstep * len(data) / m - 1
    while lastplot > 360:
        lastplot -= 360

    limit = len(gaplength)
    for i in range(len(gaplength) - 1):
        if gaplength[i, 2] > gaplength[i + 1, 2]:
            limit = i
            break

    vdata = []
    vdA = []

    for row in range(firstline - 1, lastline):
        deg = lastplot - degstep * (steps - (row - firstline + 1) // m + ignore)
        deg = deg % 360
        lvar = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(deg)
        dl = lvar / data.shape[1]
        dphi = np.pi * dK / m

        for col in range(data.shape[1]):
            val = data[row, col]
            if val != 0:
                vdata.append(val)
                vdA.append(dl * dphi)

    vdata = np.array(vdata)
    vdA = np.array(vdA)

    total_area = np.sum(vdA)

    sdata = np.sort(vdata)[::-1]  # descending sort
    x = []
    y = []

    for i in range(bins + 1):
        idx = max(0, int(np.ceil(i * len(sdata) / bins)) - 1)
        ref_val = sdata[idx]
        mask = vdata >= ref_val
        carea = np.sum(vdA[mask])
        x.append(carea / steps)
        y.append(ref_val)

    # Plot
    plt.figure()
    plt.plot(x, y, linewidth=1.5)
    plt.title('Contact Pressure vs. Cumulative Area')
    plt.xlabel('Cumulative Area [m²]')
    plt.ylabel('Contact Pressure [Pa]')
    plt.grid(True)

    out_dir = os.path.join(filepath, 'output', 'piston', 'Plots', 'Cumulative_Contact_Pressure')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'Cumulative_Contact_Pressure.png'))
    plt.close()

    print(f"Saved cumulative pressure plot to: {out_dir}")


