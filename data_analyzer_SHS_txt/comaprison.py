import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Folder path where the text files are located
folder_path = r"E:\thesis\Test-Results_Run1-Run2_speed-2500"

# File names
file_2 = 'piston_inclined-V60N_non-inclined-code.txt'
file_1 = 'piston_V60N_non-inclined-code.txt'
file_3 = 'piston_inclined-V60N_inclined-code.txt'

# File paths
file_1_path = os.path.join(folder_path, file_1)
file_2_path = os.path.join(folder_path, file_2)
file_3_path = os.path.join(folder_path, file_3)

# Read all files
data_1 = pd.read_csv(file_1_path, delimiter="\t")
data_2 = pd.read_csv(file_2_path, delimiter="\t")
data_3 = pd.read_csv(file_3_path, delimiter="\t")

# Find common columns across all three files
common_columns = data_1.columns.intersection(data_2.columns).intersection(data_3.columns)

# Calculate layout: 4 plots per page (2x2)
plots_per_page = 4
num_columns = len(common_columns)
num_pages = int(np.ceil(num_columns / plots_per_page))

# Create a PdfPages object
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = os.path.join(folder_path, 'comparison_plots.pdf')

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
                column = common_columns[plot_idx]

                # Create the plot with all three datasets using different line styles
                axes[idx].plot(data_1[column], label='Non-inclined',
                               linestyle='-',  # solid line
                               linewidth=2.5)
                axes[idx].plot(data_2[column], label='Inclined-V60N_non-inclined-code',
                               linestyle='--',  # dashed line
                               linewidth=2.5)
                axes[idx].plot(data_3[column], label='Inclined-V60N_&_code',
                               linestyle=':',  # dotted line
                               linewidth=2.5)

                axes[idx].set_title(f"Comparison of {column}", fontsize=20, pad=20)
                axes[idx].set_xlabel('Index', fontsize=16)
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