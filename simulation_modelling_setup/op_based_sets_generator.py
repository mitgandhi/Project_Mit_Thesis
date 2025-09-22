import os
import shutil
import pandas as pd
import re


def create_parameter_folders(csv_file, input_folder_source, x_folder_source):
    """
    Creates folders for each row in the CSV file and copies/modifies required files.

    Args:
        csv_file (str): Path to the CSV file with speed, pressure, and displacement values
        input_folder_source (str): Path to the source input folder with original files
        x_folder_source (str): Path to the folder X containing files to be copied
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Check if the CSV has the required columns
    required_columns = ['speed', 'presssure', 'displacement']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in the CSV file. Available columns: {df.columns.tolist()}")
            return

    # Create a new folder for each row
    for index, row in df.iterrows():
        # Extract values
        row_number = index + 1  # Row numbers typically start from 1
        speed_value = row['speed']
        pressure_value = row['presssure']
        displacement_value = row['displacement']

        # Create folder name
        folder_name = f"T_{row_number}_n_{speed_value}_dp_{pressure_value}_d_{displacement_value}"

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")

        # Create 'input' subfolder
        input_folder_path = os.path.join(folder_name, "input")
        if not os.path.exists(input_folder_path):
            os.makedirs(input_folder_path)

        # Copy all files from source input folder to the new input folder
        for file_name in os.listdir(input_folder_source):
            source_file = os.path.join(input_folder_source, file_name)
            if os.path.isfile(source_file):
                destination_file = os.path.join(input_folder_path, file_name)
                shutil.copy2(source_file, destination_file)

        # Update the operatingconditions.txt file
        update_operating_conditions(
            os.path.join(input_folder_path, "operatingconditions.txt"),
            speed_value,
            pressure_value,
            displacement_value
        )

        # Copy all files from X folder to the new parameter folder
        for file_name in os.listdir(x_folder_source):
            source_file = os.path.join(x_folder_source, file_name)
            if os.path.isfile(source_file):
                destination_file = os.path.join(folder_name, file_name)
                shutil.copy2(source_file, destination_file)

        print(f"Completed setup for folder: {folder_name}")


def update_operating_conditions(file_path, speed, pressure, displacement):
    """
    Updates the operatingconditions.txt file with the new values.

    Args:
        file_path (str): Path to the operatingconditions.txt file
        speed (int): New speed value
        pressure (int): New pressure value (HP)
        displacement (int): New displacement value (beta)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return

        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Replace the values using regular expressions
        # Update speed
        content = re.sub(r'(speed\s+)\d+', f'speed\t{speed}', content)

        # Update beta (displacement)
        content = re.sub(r'(beta\s+)\d+(\.\d+)?', f'beta\t{displacement}', content)

        # Update HP (pressure)
        content = re.sub(r'(HP\s+)\d+(\.\d+)?', f'HP\t{pressure}', content)

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.write(content)

        print(f"Updated operating conditions in {file_path}")

    except Exception as e:
        print(f"Error updating {file_path}: {str(e)}")


if __name__ == "__main__":
    # Define paths - replace these with your actual paths
    csv_file_path = "op_sets_speed_pressure_displacement.csv"
    input_folder_source_path = "./input"  # Source of input folder files
    x_folder_source_path = "./x"  # Source of folder X files

    # Create parameter folders
    create_parameter_folders(csv_file_path, input_folder_source_path, x_folder_source_path)