import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File names
file_2 = 'E:/01_CasparSims/01Parker/Mit_WithPressureDeformation_v3.1.3/P_1800_300/output/piston/piston.txt'
file_1 = 'E:/01_CasparSims/01Parker/Mit_WithPressureDeformation_v3.1.3/P_1800_300_version_3.2.0/output/piston/piston.txt'
file_3 = 'Z:/ServerSim/04Parker/Martec/final_Results/output/78_P90_1800n_300dp_100d/output/piston/piston.txt'

# File paths
file_1_path = os.path.join(file_1)
file_2_path = os.path.join(file_2)
file_3_path = os.path.join(file_3)

# Read all files
data_1 = pd.read_csv(file_1_path, delimiter="\t")
data_2 = pd.read_csv(file_2_path, delimiter="\t")
data_3 = pd.read_csv(file_3_path, delimiter="\t")

# Get the second column name (revolution) for x-axis
x_axis_column = data_1.columns[1]  # Second column (index 1)

# Find common columns across all three files
common_columns = data_1.columns.intersection(data_2.columns).intersection(data_3.columns)

# Remove the x-axis column from plotting columns (we don't want to plot revolution vs revolution)
plotting_columns = [col for col in common_columns if col != x_axis_column]

# Calculate layout: 4 plots per page (2x2)
plots_per_page = 4
num_columns = len(plotting_columns)
num_pages = int(np.ceil(num_columns / plots_per_page))

# Create a PdfPages object
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = os.path.join('comparison_plots.pdf')

with PdfPages(pdf_path) as pdf:
    # Loop through each page
    for page in range(num_pages):
        # Create a new figure for each page
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes = axes.flatten()

        # Plot up to 4 columns on this page
        for idx in range(4):
            plot_idx = page * 4 + idx
            if plot_idx < num_columns:
                column = plotting_columns[plot_idx]

                # Create the plot with revolution as x-axis and all three datasets using different line styles
                axes[idx].plot(data_1[x_axis_column], data_1[column], label='version_3.2.0',
                               linestyle='-',  # solid line
                               linewidth=2.5)
                axes[idx].plot(data_2[x_axis_column], data_2[column], label='Version_3.1.3',
                               linestyle='--',  # dashed line
                               linewidth=2.5)
                axes[idx].plot(data_3[x_axis_column], data_3[column], label='version_2_2017',
                               linestyle=':',  # dotted line
                               linewidth=2.5)

                axes[idx].set_title(f"Comparison of {column}", fontsize=20, pad=20)
                axes[idx].set_xlabel(f'{x_axis_column}', fontsize=16)  # Use revolution column name
                axes[idx].set_ylabel(column, fontsize=16)
                axes[idx].legend(fontsize=16)
                axes[idx].grid(True)
                axes[idx].tick_params(axis='both', which='major', labelsize=14)
                axes[idx].margins(x=0.05)
            else:
                # Remove unused subplots
                fig.delaxes(axes[idx])

        # Adjust layout and save page
        plt.tight_layout(pad=5.0)
        pdf.savefig(fig)
        plt.close()

print(f"PDF saved with {num_pages} pages")