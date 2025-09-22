import pandas as pd
import os

# Base path where all the Run_zeta_* folders are located
base_path = 'Z:/Studenten/Mit/Inline_Thesis-Simulation/HSP/RUN/Run4_HSP_Forces'

# Loop through Run_zeta_0 to Run_zeta_5
for i in range(10):
    folder_name = f'Run_zeta_{i}'
    input_file = os.path.join(base_path, folder_name, 'output/piston/forces_piston.txt')

    # Read the file
    df = pd.read_csv(input_file, sep='\t')

    # Add a new column as the sum of FSKy and FAKy
    df['FQ'] = df['FSKy'] + df['FAKy']

    # Format all float columns in scientific notation with 8 decimals
    float_format = '%.8e'

    # Output file name
    output_file =f'forces_piston_{i}.txt'

    # Save the modified DataFrame
    df.to_csv(output_file, sep='\t', index=False, float_format=float_format)

print("All files processed and saved.")
