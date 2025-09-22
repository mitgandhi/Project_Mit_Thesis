import os
import shutil
import csv


def modify_operating_conditions(file_path, speed):
    """Modify the speed value in operatingconditions.txt"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith('speed'):
            lines[i] = f'\tspeed\t{speed}\n'

    with open(file_path, 'w') as file:
        file.writelines(lines)


def main():
    base_dir = r"G:\Inline\Highspeed_pump\HP_VD\SIM\variable_speed"
    source_input = os.path.join(base_dir, "input")
    csv_path = os.path.join(base_dir, "speed_list.csv")

    if not os.path.exists(source_input):
        print(f"Error: Source input folder not found at {source_input}")
        return

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Read speeds from CSV
    speeds = []
    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)  # skip header if present
            for row in reader:
                if row:  # ignore empty rows
                    try:
                        speed = int(row[0])
                        speeds.append(speed)
                    except ValueError:
                        print(f"Invalid speed value: {row[0]}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Found {len(speeds)} speed values: {speeds}")

    for speed in speeds:
        folder_name = f"case_n_{speed}_"
        target_dir = os.path.join(base_dir, folder_name)
        target_input = os.path.join(target_dir, "input")

        print(f"\nCreating folder for speed {speed}: {folder_name}")
        os.makedirs(target_input, exist_ok=True)

        # Copy input template
        for item in os.listdir(source_input):
            s = os.path.join(source_input, item)
            d = os.path.join(target_input, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

        # Modify operatingconditions.txt
        conditions_file = os.path.join(target_input, "operatingconditions.txt")
        if os.path.exists(conditions_file):
            modify_operating_conditions(conditions_file, speed)
            print(f"Updated speed in: {conditions_file}")
        else:
            print(f"Warning: operatingconditions.txt not found in {target_input}")


if __name__ == "__main__":
    main()
