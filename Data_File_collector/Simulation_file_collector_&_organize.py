import os
import re
import csv

# Get the parent directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(SCRIPT_DIR, os.pardir, "Datasets")
DATASETS_PATH = os.path.abspath(DATASETS_PATH)  # Convert to absolute path

print(f"Looking for datasets in: {DATASETS_PATH}")  # Debugging line

# Check if the directory exists
if not os.path.exists(DATASETS_PATH):
    raise FileNotFoundError(f"Dataset folder not found: {DATASETS_PATH}")


def extract_parameters(filename):
    """
    Extract speed, pressure, and displacement from filename.
    Example filename: V60N_S4_dp_350_n_2000_d_100.txt
    Returns: (speed, pressure, displacement)
    """
    match = re.search(r"dp_(\d+)_n_(\d+)_d_(\d+)", filename)
    if match:
        pressure, speed, displacement = match.groups()
        return int(speed), int(pressure), int(displacement)
    return None, None, None


def rename_files():
    """
    Iterate over DSP folders and rename files.
    """
    for dsp_folder in os.listdir(DATASETS_PATH):
        if (dsp_folder != "DS"):
            dsp_path = os.path.join(DATASETS_PATH, dsp_folder)
            if not os.path.isdir(dsp_path):
                continue  # Skip non-directory files

            for filename in os.listdir(dsp_path):
                if filename.endswith(".txt"):
                    speed, pressure, displacement = extract_parameters(filename)

                    if speed is not None:
                        # Rename file
                        new_filename = f"piston_{speed}_{pressure}_{displacement}.txt"
                        old_filepath = os.path.join(dsp_path, filename)
                        new_filepath = os.path.join(dsp_path, new_filename)

                        os.rename(old_filepath, new_filepath)
                        print(f"Renamed: {filename} → {new_filename}")

            print("All files have been renamed successfully.")
        else:
            continue




def generate_csv():
    """
    Iterate over DSP folders inside /DS and create CSV files in the same directory.
    """
    DSP_ROOT = os.path.join(DATASETS_PATH, "DS")  # DS folder inside DATASETS_PATH
    print(DSP_ROOT)
    if not os.path.exists(DSP_ROOT):
        raise FileNotFoundError(f"DSP root directory not found: {DSP_ROOT}")

    for dsp_folder in os.listdir(DATASETS_PATH):
        dsp_path = os.path.join(DATASETS_PATH, dsp_folder)
        if not os.path.isdir(dsp_path):
            continue  # Skip non-directory files

        # Save CSV inside the same DS folder
        csv_filename = os.path.join(DSP_ROOT, f"{dsp_folder}.csv")

        with open(csv_filename, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["ID", "speed", "pressure", "displacement"])  # CSV header

            pd_index = 1  # Counter for PD1, PD2, ...

            for filename in os.listdir(dsp_path):
                if filename.startswith("piston_") and filename.endswith(".txt"):
                    parts = filename.replace("piston_", "").replace(".txt", "").split("_")
                    if len(parts) == 3:
                        try:
                            speed, pressure, displacement = map(int, parts)

                            # Write to CSV
                            csv_writer.writerow([f"PD{pd_index}", speed, pressure, displacement])
                            pd_index += 1
                        except ValueError:
                            print(f"Skipping file with incorrect format: {filename}")

        print(f"CSV file created: {csv_filename}")


if __name__ == "__main__":
    choice = input("Enter 'rename' to rename files or 'csv' to generate CSVs: ").strip().lower()

    if choice == "rename":
        rename_files()
    elif choice == "csv":
        generate_csv()
    else:
        print("Invalid choice! Please enter 'rename' or 'csv'.")
