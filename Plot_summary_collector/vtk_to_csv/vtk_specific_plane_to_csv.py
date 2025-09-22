import os
import pyvista as pv
import numpy as np
import pandas as pd

# Root directory where all simulation folders exist
root_directory = "Z:/ServerSim/04Parker/Martec/SimRun18_slipperWear"  # Change to your actual path

# Define expected time steps
timesteps = [0, 360, 720, 1080, 1440, 1800,2160,2520,2840,3240]
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

# Z-coordinate for the plane
plane_z = -0.0005  # 0.5mm depth as specified

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

                    # Extract temperature at the specified plane
                    # Create a plane at z = plane_z with normal (0,0,1)
                    bounds = mesh.bounds
                    plane = pv.Plane(
                        center=(
                            (bounds[0] + bounds[1]) / 2,  # x center
                            (bounds[2] + bounds[3]) / 2,  # y center
                            plane_z  # specified z position
                        ),
                        direction=(0, 0, 1),  # normal direction
                        i_size=(bounds[1] - bounds[0]) * 1.5,  # make plane slightly larger than mesh
                        j_size=(bounds[3] - bounds[2]) * 1.5
                    )

                    # Sample the data onto the plane
                    plane_data = mesh.sample(plane)

                    if "temperature" in plane_data.point_data:
                        plane_temp = plane_data.point_data["temperature"]
                        plane_min = np.min(plane_temp)
                        plane_mean = np.mean(plane_temp)
                        plane_max = np.max(plane_temp)

                        # Store plane temperature results
                        results[sim_name][f"plane_min_{step}"] = plane_min
                        results[sim_name][f"plane_mean_{step}"] = plane_mean
                        results[sim_name][f"plane_max_{step}"] = plane_max

                        print(f"  ✅ Plane temperature data at z={plane_z}:")
                        print(f"     min={plane_min:.2f}, mean={plane_mean:.2f}, max={plane_max:.2f}")
                    else:
                        print(f"  ❌ No temperature data sampled on the plane")
                        results[sim_name][f"plane_min_{step}"] = np.nan
                        results[sim_name][f"plane_mean_{step}"] = np.nan
                        results[sim_name][f"plane_max_{step}"] = np.nan
                else:
                    print(f"  ❌ Missing 'temperature' field. Available fields: {available_fields}")
                    results[sim_name][f"min_{step}"] = np.nan
                    results[sim_name][f"mean_{step}"] = np.nan
                    results[sim_name][f"max_{step}"] = np.nan
                    results[sim_name][f"plane_min_{step}"] = np.nan
                    results[sim_name][f"plane_mean_{step}"] = np.nan
                    results[sim_name][f"plane_max_{step}"] = np.nan
            except Exception as e:
                print(f"  ❌ Error reading {vtk_file}: {e}")
                results[sim_name][f"min_{step}"] = np.nan
                results[sim_name][f"mean_{step}"] = np.nan
                results[sim_name][f"max_{step}"] = np.nan
                results[sim_name][f"plane_min_{step}"] = np.nan
                results[sim_name][f"plane_mean_{step}"] = np.nan
                results[sim_name][f"plane_max_{step}"] = np.nan
        else:
            print(f"  ❌ File not found: {vtk_file}")
            results[sim_name][f"min_{step}"] = np.nan
            results[sim_name][f"mean_{step}"] = np.nan
            results[sim_name][f"max_{step}"] = np.nan
            results[sim_name][f"plane_min_{step}"] = np.nan
            results[sim_name][f"plane_mean_{step}"] = np.nan
            results[sim_name][f"plane_max_{step}"] = np.nan

# Convert to DataFrame
if results:
    print("\nCreating DataFrame...")
    df = pd.DataFrame.from_dict(results, orient='index')

    if not df.empty:
        # Sort columns by timestep
        df = df.reindex(sorted(df.columns, key=lambda x: (
        x.split("_")[0], int(x.split("_")[1]) if len(x.split("_")) > 1 and x.split("_")[1].isdigit() else 0)), axis=1)

        # Save to CSV
        output_file = "temperature_summary_with_plane.csv"
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