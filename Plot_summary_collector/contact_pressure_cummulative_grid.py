import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import savemat
import warnings
import csv

warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_matlab_txt(path):
    return np.genfromtxt(path, comments='%', delimiter=None)

def round2(x, base=0.6):
    return base * round(x / base)

def piston_contact_pressure(filepath, m, lF, n, deltap, plots, degstep, ognore, minplot, maxplot, offset):

    ignore = ognore + offset
    data = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt'))
    gaplength = load_matlab_txt(os.path.join(filepath, 'output', 'piston', 'piston.txt'))

    xdata = np.linspace(0, 360, m)
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
        if gaplength[i, 2] > gaplength[i + 1, 2]:
            limit = i
            break

    out_dir = os.path.join(filepath, 'output', 'piston', 'Plots', 'Piston_Contact_Pressure_cummulativ_grid')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGenerating cumulative contact pressure plot for: {filepath}")

    cumulative_frame = np.zeros((m, data.shape[1]))
    for i in range(plots):
        frame = data[firstline - 1 + i * m: firstline - 1 + (i + 1) * m, :]
        cumulative_frame += frame

    mid_deg = degstep * (plots // 2 + ignore) % 360
    interp_len = interp1d(gaplength[:limit, 2], gaplength[:limit, 11], fill_value='extrapolate')(mid_deg)
    interp_off = interp1d(gaplength[:limit, 2], gaplength[:limit, 18], fill_value='extrapolate')(mid_deg)
    ydata = np.linspace(0, interp_len, data.shape[1]) + interp_off

    # Cumulative Pressure Plot
    plt.figure()
    c = plt.pcolormesh(ydata, xdata, cumulative_frame, shading='auto', cmap='hot')
    plt.title(f'Cumulative Contact Pressure\nn={n}, ΔP={deltap}')
    plt.xlabel('Gap Length [m]')
    plt.ylabel('Gap Circumference [degrees]')
    plt.xticks(np.arange(0, lF * 1e-3 + 1e-5, 0.01))
    try:
        plt.colorbar(c, label='Cumulative Pressure [Pa]')
    except Exception as e:
        print(f"Warning: Skipping colorbar due to error: {e}")
    plt.yticks(np.arange(0, 361, 60))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Cumulative_Contact_Pressure_Only_One_Plot.png'))
    plt.close()

    print("\n✅ Single cumulative plot generated and saved.")

    # Average Pressure Map
    average_frame = cumulative_frame / plots
    plt.figure()
    plt.pcolormesh(ydata, xdata, average_frame, shading='auto', cmap='hot')
    plt.title('Time-Averaged Contact Pressure')
    plt.xlabel('Gap Length [m]')
    plt.ylabel('Gap Circumference [degrees]')
    try:
        plt.colorbar(label='Average Pressure [Pa]')
    except Exception as e:
        print(f"Warning: Skipping colorbar for average pressure plot: {e}")
    plt.xticks(np.arange(0, lF * 1e-3 + 1e-5, 0.01))
    plt.yticks(np.arange(0, 361, 60))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Average_Contact_Pressure.png'))
    plt.close()

    # Statistics
    mean_pressure = np.mean(cumulative_frame)
    max_pressure = np.max(cumulative_frame)
    min_pressure = np.min(cumulative_frame)
    std_pressure = np.std(cumulative_frame)

    mean_over_circumference = np.mean(cumulative_frame, axis=0)
    max_over_circumference = np.max(cumulative_frame, axis=0)
    mean_over_length = np.mean(cumulative_frame, axis=1)
    max_over_length = np.max(cumulative_frame, axis=1)

    print("\n--- Cumulative Pressure Statistics ---")
    print(f"Mean Pressure: {mean_pressure:.2f} Pa")
    print(f"Max Pressure: {max_pressure:.2f} Pa")
    print(f"Min Pressure: {min_pressure:.2f} Pa")
    print(f"Standard Deviation: {std_pressure:.2f} Pa")

    stats_csv_path = os.path.join(out_dir, 'Cumulative_Pressure_Stats.csv')
    with open(stats_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value (Pa)'])
        writer.writerow(['Mean Pressure', mean_pressure])
        writer.writerow(['Max Pressure', max_pressure])
        writer.writerow(['Min Pressure', min_pressure])
        writer.writerow(['Standard Deviation', std_pressure])
        writer.writerow([])
        writer.writerow(['Gap Length Index', 'Mean Pressure', 'Max Pressure'])
        for i in range(len(mean_over_circumference)):
            writer.writerow([i, mean_over_circumference[i], max_over_circumference[i]])
        writer.writerow([])
        writer.writerow(['Circumference Index', 'Mean Pressure', 'Max Pressure'])
        for i in range(len(mean_over_length)):
            writer.writerow([i, mean_over_length[i], max_over_length[i]])

    print(f"Statistics saved to CSV: {stats_csv_path}\n")

    # Hotspot Detection
    threshold = 0.8 * max_pressure
    hotspot_mask = cumulative_frame >= threshold

    plt.figure()
    plt.pcolormesh(ydata, xdata, cumulative_frame, shading='auto', cmap='hot')
    plt.contour(ydata, xdata, hotspot_mask, levels=[0.5], colors='cyan', linewidths=1.0)
    plt.title('Hotspots on Cumulative Pressure Map')
    plt.xlabel('Gap Length [m]')
    plt.ylabel('Gap Circumference [degrees]')
    try:
        plt.colorbar(label='Cumulative Pressure [Pa]')
    except Exception as e:
        print(f"Warning: Skipping colorbar for hotspot overlay: {e}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Hotspots_Overlay.png'))
    plt.close()

    plt.figure()
    plt.pcolormesh(ydata, xdata, hotspot_mask, shading='auto', cmap='gray')
    plt.title('Hotspot Mask')
    plt.xlabel('Gap Length [m]')
    plt.ylabel('Gap Circumference [degrees]')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Hotspot_Mask.png'))
    plt.close()

    # Line Plots every 5 degrees from 0 to 355
    plt.figure()
    for deg in range(0, 360, 5):
        idx = int(m * deg / 360)
        plt.plot(ydata, cumulative_frame[idx, :], label=f"{deg}°")

    plt.title('Cumulative Pressure vs. Gap Length (every 5°)')
    plt.xlabel('Gap Length [m]')
    plt.ylabel('Cumulative Pressure [Pa]')
    plt.grid(True)
    plt.legend(title='Circumference', fontsize='x-small', ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Pressure_Lines_0_to_360_by_5deg.png'))
    plt.close()

    # Export to MAT
    savemat(os.path.join(out_dir, 'cumulative_frame.mat'), {
        'cumulative_frame': cumulative_frame,
        'ydata': ydata,
        'xdata': xdata,
        'mean_over_circumference': mean_over_circumference,
        'max_over_circumference': max_over_circumference,
        'mean_over_length': mean_over_length,
        'max_over_length': max_over_length
    })

    print("\n✅ Additional analysis completed and saved, including MAT and CSV export.")
