import pandas as pd
import numpy as np

file_path = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run6_ReDimensions\Terst_bench_test_4500_320bar\T01_dZ_19.5017_dK_19.4641\output\piston\piston.txt'

try:
    # Skip the comment row and load the actual header
    df = pd.read_csv(file_path, sep='\t', skiprows=1, engine='python')
    df.columns = df.columns.str.strip()

    # Ensure required columns exist
    if 'shaft_angle' in df.columns and 'Total_Leakage' in df.columns:
        # Optional: convert shaft_angle to numeric if not already
        df['shaft_angle'] = pd.to_numeric(df['shaft_angle'], errors='coerce')
        df['Total_Leakage'] = pd.to_numeric(df['Total_Leakage'], errors='coerce')

        # Drop rows with NaN values (if any)
        df.dropna(subset=['shaft_angle', 'Total_Leakage'], inplace=True)

        # Create a new column for revolution number (0, 1, 2, ...)
        df['revolution'] = (df['shaft_angle'] // 360).astype(int)

        # Group by revolution and sum the leakage per 360°
        leakage_per_rev = df.groupby('revolution')['Total_Leakage'].sum()

        # Print the results
        print("\nTotal Leakage per 360° revolution:")
        for rev, leak in leakage_per_rev.items():
            print(f"Revolution {rev}: {leak:.6e}")

    else:
        print("Required columns not found in the file.")
        print("Available columns:", df.columns.tolist())

except FileNotFoundError:
    print(f"File not found: {file_path}")
