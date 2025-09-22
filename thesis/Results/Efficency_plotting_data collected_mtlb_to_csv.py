import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Base folder containing all CSV files
base_folder = 'G:\\Inline\\Highspeed_pump\\Results\\T01_Efficency'

# Get all CSV files in the folder (non-recursive)
csv_files = glob.glob(os.path.join(base_folder, '*.csv'))

# Load DataFrames and labels
dataframes = []
labels = []

def clean_label(path):
    name = os.path.basename(path)
    return name.replace('Simulation_efficecny_results_', '').replace('.csv', '')

for path in csv_files:
    df = pd.read_csv(path, header=None)
    # Clean specific known file (adjust condition as needed)
    if 'straight_piston' in path:
        df = df[df.iloc[:, 1].astype(str).str.isdigit()].copy()
        df.iloc[:, 1] = df.iloc[:, 1].astype(int)
    dataframes.append(df)
    labels.append(clean_label(path))

# Plot function
def plot_efficiency(col_index, efficiency_name):
    # Create dictionary for each DataFrame
    value_dicts = [dict(zip(df.iloc[:, 1], df.iloc[:, col_index])) for df in dataframes]

    # Use x-values from the first file
    x_values = dataframes[0].iloc[:, 1].values
    y_values_list = [[d.get(x, None) for x in x_values] for d in value_dicts]

    # Filter out rows where any y is None
    filtered_data = [
        (x, *[y[i] for y in y_values_list])
        for i, x in enumerate(x_values)
        if all(y[i] is not None for y in y_values_list)
    ]

    if not filtered_data:
        print(f"No common x-values found for {efficiency_name}")
        return

    x_final, *ys_final = zip(*filtered_data)

    # Plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', len(dataframes))  # Use a color map

    for i, (y_vals, label) in enumerate(zip(ys_final, labels)):
        plt.plot(x_final, y_vals, label=label, linestyle='-')
        plt.scatter(x_final, y_vals, color=colors(i))

    plt.xlabel('Speed (RPM)')
    plt.ylabel(f'{efficiency_name} (%)')
    plt.title(f'Speed vs. {efficiency_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot all three efficiency types
plot_efficiency(-1, 'Overall Efficiency')
plot_efficiency(-2, 'Hydro-Mechanical Efficiency')
plot_efficiency(-3, 'Volumetric Efficiency')
