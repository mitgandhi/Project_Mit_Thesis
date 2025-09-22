import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import math

def create_comparison_plots(file_1_path, file_2_path, output_pdf="comparison_plots.pdf"):
    # Read data files
    data_1 = pd.read_csv(file_1_path, delimiter="\t")
    data_2 = pd.read_csv(file_2_path, delimiter="\t")

    # Get common columns
    common_columns = data_1.columns.intersection(data_2.columns)

    # Calculate number of pages needed (4 plots per row, 2 rows per page)
    plots_per_page = 8
    num_pages = math.ceil(len(common_columns) / plots_per_page)

    with PdfPages(output_pdf) as pdf:
        for page in range(num_pages):
            # Create figure with subplots (2 rows, 4 columns)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)

            # Get columns for current page
            start_idx = page * plots_per_page
            page_columns = common_columns[start_idx:start_idx + plots_per_page]

            # Create plots for current page
            for idx, column in enumerate(page_columns):
                row = idx // 4
                col = idx % 4

                axes[row, col].plot(data_1[column], label='173_piston')
                axes[row, col].plot(data_2[column], label='local_device')
                axes[row, col].set_title(column)
                axes[row, col].set_xlabel('Index')
                axes[row, col].set_ylabel(column)
                axes[row, col].legend()
                axes[row, col].grid(True)

            # Hide empty subplots if any
            for idx in range(len(page_columns), plots_per_page):
                row = idx // 4
                col = idx % 4
                axes[row, col].set_visible(False)

            # Add page title
            fig.suptitle(f'Piston Data Comparison - Page {page + 1}', fontsize=16)

            # Save current page
            pdf.savefig()
            plt.close()

# Example usage
folder_path = "/"
file_1 = "../slipper_173-version_3.1.3-option3-withFslipper.txt"
file_2 = "../slipper_188.txt"

file_1_path = os.path.join(folder_path, file_1)
file_2_path = os.path.join(folder_path, file_2)

create_comparison_plots(file_1_path, file_2_path)