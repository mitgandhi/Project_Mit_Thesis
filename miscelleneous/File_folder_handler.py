import os
import shutil


def ignore_output_folder(directory, files):
    """Function to ignore the 'output' folder while copying."""
    return ["output"] if "output" in files else []


def create_backup_and_update_files(source_root, backup_root, new_im_piston_path, new_mesh_file_path, Dia, L):
    """
    Step 1: Copy all subdirectories (excluding 'output') to a new backup location.
    Step 2: Modify paths in 'options_piston.txt' and 'thermal_piston.txt' in the new location.

    :param source_root: The original directory containing multiple subfolders.
    :param backup_root: The directory where the subfolders will be copied.
    :param new_im_piston_path: The new path for IM_piston in options_piston.txt.
    :param new_mesh_file_path: The new path for meshFile in thermal_piston.txt.
    """

    if not os.path.exists(backup_root):
        os.makedirs(backup_root)

    for subdir in os.listdir(source_root):
        source_subdir_path = os.path.join(source_root, subdir)
        backup_subdir_path = os.path.join(backup_root, subdir)

        if os.path.isdir(source_subdir_path) and subdir.lower() != "output":  # Exclude 'output' folder
            # Copy the entire subdirectory excluding the 'output' folder
            shutil.copytree(source_subdir_path, backup_subdir_path, dirs_exist_ok=True, ignore=ignore_output_folder)
            print(f"✅ Copied {source_subdir_path} -> {backup_subdir_path}")

            # Path to input folder in the copied location
            input_folder = os.path.join(backup_subdir_path, "input")

            if os.path.exists(input_folder):
                geometry_piston_file = os.path.join(input_folder,"geometry.txt")
                options_piston_file = os.path.join(input_folder, "options_piston.txt")
                thermal_piston_file = os.path.join(input_folder, "thermal_piston.txt")

                # Update IM_piston_path in options_piston.txt
                if os.path.exists(options_piston_file):
                    with open(options_piston_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    modified = False
                    for i, line in enumerate(lines):
                        if "IM_piston_path" in line:
                            print(f"🔄 Replacing IM_piston_path in {options_piston_file}")
                            lines[i] = f"    IM_piston_path\t{new_im_piston_path}\n"
                            modified = True

                    if modified:
                        with open(options_piston_file, "w", encoding="utf-8") as f:
                            f.writelines(lines)
                        print(f"✅ Updated: {options_piston_file}")

                # Update meshFile in thermal_piston.txt
                if os.path.exists(thermal_piston_file):
                    with open(thermal_piston_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    modified = False
                    for i, line in enumerate(lines):
                        if "meshFile" in line:
                            print(f"🔄 Replacing meshFile in {thermal_piston_file}")
                            lines[i] = f"meshFile\t{new_mesh_file_path}\n"
                            modified = True

                    if modified:
                        with open(thermal_piston_file, "w", encoding="utf-8") as f:
                            f.writelines(lines)
                        print(f"✅ Updated: {thermal_piston_file}")
                    else:
                        print(f"⚠️ No changes made to {thermal_piston_file}. Check the file content.")

                # Chaning the diameter of in input
                if os.path.exists(geometry_piston_file):
                    with open(geometry_piston_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    modified = False
                    for i, line in enumerate(lines):
                        if "dK" in line and "speedK" not in line:
                            print(f"🔄 Replacing dK in {geometry_piston_file}")
                            lines[i] = f"\tdK\t{Dia}\n"
                            modified = True

                        if "lK" in line and "lK_hs" not in line and "lKG" not in line:
                            print(f"🔄 Replacing lK in {geometry_piston_file}")
                            lines[i] = f"\tlK\t{L}\n"
                            modified = True

                    if modified:
                        with open(geometry_piston_file, "w", encoding="utf-8") as f:
                            f.writelines(lines)
                        print(f"✅ Updated: {geometry_piston_file}")


# Paths
Diameter = 20.52
Length = 70.1
source_root = "Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run1_Test_V60N\Run_V60N"
backup_root = f"Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run1_Test_V60N\Run_V60N_D{Diameter}_L{Length}"
new_im_piston_path = f"E:/01_CasparSims/02Inline/IM_S4/pistoncylinder/IM_piston_D{Diameter}_L{Length}"
new_mesh_file_path = f"E:/01_CasparSims/02Inline/IM_S4/pistoncylinder/thermal/piston_th_D{Diameter}_L{Length}.inp"

# Run the function
create_backup_and_update_files(source_root, backup_root, new_im_piston_path, new_mesh_file_path, Diameter, Length)
