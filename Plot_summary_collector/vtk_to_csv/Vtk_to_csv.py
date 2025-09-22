import os
import pyvista as pv
import numpy as np
import pandas as pd

# Root directory where all simulation folders exist
root_directory = "Z:/ServerSim/04Parker/Martec/final_Results/output"  # Change to your actual path

# Define expected time steps
timesteps = [0, 360, 720, 1080, 1440]
file_prefix = "swashplate_thermal"
file_suffix = ".vtk"

# Store results: {simulation_folder_name: {stat_step: value}}
results = {}

# Diagnostic counters
counters = {
    "total_dirs": 0,
    "valid_sim_dirs": 0,
    "valid_vtk_folders": 0,
    "found_vtk_files": 0,
    "valid_temp_data": 0
}

# Loop through all subdirectories under root
print(f"Checking root directory: {root_directory}")
if not os.path.exists(root_directory):
    print(f"❌ Root directory does not exist: {root_directory}")
    exit()

subdirs = os.listdir(root_directory)
print(f"Found {len(subdirs)} items in root directory")
counters["total_dirs"] = len(subdirs)

for sim_name in subdirs:
    sim_path = os.path.join(root_directory, sim_name)

    if not os.path.isdir(sim_path):
        print(f"Skipping non-directory: {sim_name}")
        continue

    counters["valid_sim_dirs"] += 1
    print(f"\nProcessing simulation directory: {sim_name}")

    # Path to the expected vtk_saved folder
    vtk_folder = os.path.join(sim_path, "output", "slipper", "vtk")

    if not os.path.isdir(vtk_folder):
        print(f"❌ VTK folder not found: {vtk_folder}")
        continue

    counters["valid_vtk_folders"] += 1
    print(f"✅ Found VTK folder: {vtk_folder}")

    results[sim_name] = {}

    for step in timesteps:
        vtk_file = os.path.join(vtk_folder, f"{file_prefix}.{step}{file_suffix}")

        if os.path.exists(vtk_file):
            print(f"  Processing file: {os.path.basename(vtk_file)}")
            counters["found_vtk_files"] += 1

            try:
                mesh = pv.read(vtk_file)
                available_fields = list(mesh.point_data.keys())
                print(f"  Available data fields: {available_fields}")

                temp_data = mesh.point_data.get("temperature")

                if temp_data is not None:
                    counters["valid_temp_data"] += 1
                    min_val = np.min(temp_data)
                    mean_val = np.mean(temp_data)
                    max_val = np.max(temp_data)

                    results[sim_name][f"min_{step}"] = min_val
                    results[sim_name][f"mean_{step}"] = mean_val
                    results[sim_name][f"max_{step}"] = max_val

                    print(f"  ✅ Temperature data found: min={min_val:.2f}, mean={mean_val:.2f}, max={max_val:.2f}")
                else:
                    print(f"  ❌ Missing 'temperature' field. Available fields: {available_fields}")
                    results[sim_name][f"min_{step}"] = np.nan
                    results[sim_name][f"mean_{step}"] = np.nan
                    results[sim_name][f"max_{step}"] = np.nan
            except Exception as e:
                print(f"  ❌ Error reading {vtk_file}: {e}")
                results[sim_name][f"min_{step}"] = np.nan
                results[sim_name][f"mean_{step}"] = np.nan
                results[sim_name][f"max_{step}"] = np.nan
        else:
            print(f"  ❌ File not found: {vtk_file}")
            results[sim_name][f"min_{step}"] = np.nan
            results[sim_name][f"mean_{step}"] = np.nan
            results[sim_name][f"max_{step}"] = np.nan

# Convert to DataFrame
if results:
    print("\nCreating DataFrame...")
    df = pd.DataFrame.from_dict(results, orient='index')

    if not df.empty:
        # Sort columns by timestep
        df = df.reindex(sorted(df.columns, key=lambda x: (x.split("_")[0], int(x.split("_")[1]))), axis=1)

        # Save to CSV
        output_file = "temperature_summary.csv"
        df.to_csv(output_file)
        print(f"\n✅ Summary saved to: {output_file}")
        print(f"DataFrame shape: {df.shape}")
    else:
        print("\n❌ Generated DataFrame is empty!")
else:
    print("\n❌ No results were collected!")

# Print summary
print("\n--- SUMMARY ---")
print(f"Total directories in root: {counters['total_dirs']}")
print(f"Valid simulation directories: {counters['valid_sim_dirs']}")
print(f"Valid VTK folders found: {counters['valid_vtk_folders']}")
print(f"VTK files found: {counters['found_vtk_files']}")
print(f"Files with valid temperature data: {counters['valid_temp_data']}")