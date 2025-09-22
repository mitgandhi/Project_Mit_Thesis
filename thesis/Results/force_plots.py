import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate

# Base paths for the two folders
FOLDER1 = r"G:\Inline\Highspeed_pump\HP_VD\results\csv\Inclination_CSV_Last_Revolution"
FOLDER2 = r"G:\Inline\Highspeed_pump\HP_VD\results\csv\inclincation_csv_with_streamlit"

# Specific files to compare (gamma_0, gamma_5, gamma_10, gamma_15)
GAMMA_FILES = ['gamma_0', 'gamma_5', 'gamma_10', 'gamma_15']

# Columns to plot
COLUMNS_TO_PLOT = ['FSKy', 'FAKy', 'Querkraft']


def load_files():
    """Load all specified gamma files from both folders."""
    file_pairs = {}
    for gamma_file in GAMMA_FILES:
        file1 = os.path.join(FOLDER1, f"{gamma_file}.csv")
        file2 = os.path.join(FOLDER2, f"{gamma_file}_data.csv")

        if os.path.exists(file1) and os.path.exists(file2):
            file_pairs[gamma_file] = (file1, file2)
        else:
            if not os.path.exists(file1):
                print(f"Warning: {file1} not found")
            if not os.path.exists(file2):
                print(f"Warning: {file2} not found")

    return file_pairs


def load_and_prep_data(file_path1, file_path2):
    """Load and prepare both CSV files for comparison."""
    print(f"Loading data from:\n{file_path1}\n{file_path2}")
    try:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)

        print(f"Successfully loaded both files:")
        print(f"- File 1: {len(df1)} rows, {len(df1.columns)} columns")
        print(f"- File 2: {len(df2)} rows, {len(df2.columns)} columns")

        column_mapping = {
            'phi': 'shaft_angle',
            'QuerKraft': 'Querkraft'
        }
        df2 = df2.rename(columns=column_mapping)

        return df1, df2
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def create_split_combined_plots(file_pairs, output_dir):
    """Create plots split into 0–200 and 200–360 shaft angle ranges with dynamic y-axis scaling."""
    for column_name in COLUMNS_TO_PLOT:
        processed_data = {}

        for gamma_file, (file1, file2) in file_pairs.items():
            df1, df2 = load_and_prep_data(file1, file2)
            if df1 is None or df2 is None:
                continue

            column_name2 = column_name
            if column_name == 'Querkraft' and 'QuerKraft' in df2.columns:
                column_name2 = 'QuerKraft'

            processed_data[gamma_file] = {
                'df1': df1,
                'df2': df2,
                'column_name1': column_name,
                'column_name2': column_name2
            }

        line_styles = {
            'folder1': {
                'gamma_0': {'linestyle': 'dotted', 'color': 'blue', 'linewidth': 2},
                'gamma_5': {'linestyle': 'dotted', 'color': 'green', 'linewidth': 2},
                'gamma_10': {'linestyle': 'dotted', 'color': 'red', 'linewidth': 2},
                'gamma_15': {'linestyle': 'dotted', 'color': 'purple', 'linewidth': 2}
            },
            'folder2': {
                'gamma_0': {'linestyle': 'dashed', 'color': 'blue', 'linewidth': 2},
                'gamma_5': {'linestyle': 'dashed', 'color': 'green', 'linewidth': 2},
                'gamma_10': {'linestyle': 'dashed', 'color': 'red', 'linewidth': 2},
                'gamma_15': {'linestyle': 'dashed', 'color': 'purple', 'linewidth': 2}
            }
        }

        for angle_range, title_suffix in [((0, 200), "0-200°"), ((200, 360), "200-360°")]:
            angle_min, angle_max = angle_range
            range_values = []

            for gamma_file, data in processed_data.items():
                df1 = data['df1']
                df2 = data['df2']
                col1 = data['column_name1']
                col2 = data['column_name2']
                angle_col1 = 'shaft_angle'
                angle_col2 = 'shaft_angle' if 'shaft_angle' in df2.columns else 'phi'

                df1_range = df1[(df1[angle_col1] >= angle_min) & (df1[angle_col1] <= angle_max)]
                df2_range = df2[(df2[angle_col2] >= angle_min) & (df2[angle_col2] <= angle_max)]

                if col1 in df1_range.columns:
                    range_values.extend(df1_range[col1].dropna().tolist())
                if col2 in df2_range.columns:
                    range_values.extend(df2_range[col2].dropna().tolist())

            if range_values:
                y_min = min(range_values)
                y_max = max(range_values)
                if y_min == y_max:
                    y_min -= 0.1 * abs(y_min) if y_min != 0 else -1
                    y_max += 0.1 * abs(y_max) if y_max != 0 else 1
            else:
                y_min, y_max = -1, 1

            plt.figure(figsize=(15, 8))

            for gamma_file, data in processed_data.items():
                df1 = data['df1']
                df2 = data['df2']
                col1 = data['column_name1']
                col2 = data['column_name2']
                angle_col1 = 'shaft_angle'
                angle_col2 = 'shaft_angle' if 'shaft_angle' in df2.columns else 'phi'

                df1_range = df1[(df1[angle_col1] >= angle_min) & (df1[angle_col1] <= angle_max)].copy()
                df2_range = df2[(df2[angle_col2] >= angle_min) & (df2[angle_col2] <= angle_max)].copy()

                df1_range = df1_range.sort_values(by=angle_col1)
                df2_range = df2_range.sort_values(by=angle_col2)

                if col1 in df1.columns:
                    plt.plot(df1_range[angle_col1], df1_range[col1],
                             **line_styles['folder1'][gamma_file],
                             label=f'{gamma_file} (Folder 1)')

                if col2 in df2.columns:
                    plt.plot(df2_range[angle_col2], df2_range[col2],
                             **line_styles['folder2'][gamma_file],
                             label=f'{gamma_file}_data (Folder 2)')

            plt.title(f"Combined {column_name} Comparison - {title_suffix}", fontsize=16)
            plt.xlabel('Shaft Angle (degrees)')
            plt.ylabel(f'{column_name} Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(angle_min, angle_max)
            plt.axvline(x=angle_min, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=angle_max, color='gray', linestyle='--', alpha=0.5)
            plt.ylim(y_min, y_max)

            plot_filename = f"combined_{column_name}_comparison_{angle_min}-{angle_max}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"Created combined plot for {title_suffix}: {plot_path}")


def main():
    output_dir = "comparison_plots"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output will be saved to: {os.path.abspath(output_dir)}")

    file_pairs = load_files()
    if not file_pairs:
        print("No matching file pairs found. Check file paths.")
        return

    print(f"Found {len(file_pairs)} file pairs to compare:")
    for gamma_file, (file1, file2) in file_pairs.items():
        print(f"- {gamma_file}: \n  {file1} \n  {file2}")

    create_split_combined_plots(file_pairs, output_dir)
    print("\nAll plots have been created successfully.")


if __name__ == "__main__":
    main()
