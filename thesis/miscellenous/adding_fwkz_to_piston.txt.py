import pandas as pd
import numpy as np
import os
import glob
import re


def process_piston_files(folder_path):
    """
    Process all piston files in a folder and add FwKz column.

    Args:
        folder_path (str): Path to the folder containing piston files
    """
    # Find all piston files matching the pattern
    file_pattern = os.path.join(folder_path, "piston_*.txt")
    piston_files = glob.glob(file_pattern)

    if not piston_files:
        print(f"No piston files found in {folder_path}")
        return

    print(f"Found {len(piston_files)} piston files:")

    for file_path in piston_files:
        # Extract filename without path and extension
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")

        # Extract angle from filename using regex
        # Pattern matches: piston_0.txt, piston_5.txt, piston_10.txt, piston_15.txt
        match = re.search(r'piston_(\d+)\.txt', filename)

        if not match:
            print(f"  Warning: Could not extract angle from {filename}")
            continue

        angle_degrees = int(match.group(1))

        # Convert angle to radians
        angle_radians = np.radians(angle_degrees)

        try:
            # Read the data file
            # Using tab separator and first row as header
            df = pd.read_csv(file_path, sep='\t')

            # Check if FwK column exists
            if 'FwK' not in df.columns:
                print(f"  Warning: FwK column not found in {filename}")
                print(f"  Available columns: {list(df.columns)}")
                continue

            # Calculate FwKz = FwK * sin(angle)
            df['Fwkz'] = df['FwK'] * np.sin(angle_radians)

            # Save the modified dataframe back to the original file
            df.to_csv(file_path, sep='\t', index=False)

            print(f"  Added FwKz column with angle {angle_degrees}° (sin = {np.sin(angle_radians):.6f})")
            print(f"  Original file updated: {filename}")
            print(f"  Rows processed: {len(df)}")

        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")

        print()  # Empty line for readability


def preview_file(file_path, num_rows=3):
    """
    Preview the first few rows of a processed file.

    Args:
        file_path (str): Path to the file
        num_rows (int): Number of rows to display
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Preview of {os.path.basename(file_path)}:")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head(num_rows).to_string(index=False))
        print(f"\nFwkz statistics:")
        print(f"  Min: {df['Fwkz'].min():.6f}")
        print(f"  Max: {df['Fwkz'].max():.6f}")
        print(f"  Mean: {df['Fwkz'].mean():.6f}")
    except Exception as e:
        print(f"Error previewing file: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Set your folder path here
    folder_path = "C:/Users/MIT/Desktop/thesis_temp/Results_inclined_piston_forces_caspar/Results/literature"  # Current directory, change as needed

    print("Piston Data Processor")
    print("=" * 50)

    # Process all piston files
    process_piston_files(folder_path)

    # Preview one of the original files (if it exists)
    sample_file = os.path.join(folder_path, "piston_0.txt")
    if os.path.exists(sample_file):
        print("\n" + "=" * 50)
        print("PREVIEW OF PROCESSED FILE")
        print("=" * 50)
        preview_file(sample_file)