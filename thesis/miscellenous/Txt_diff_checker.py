import pandas as pd


def subtract_txt_files(file1, file2, output_file):
    # Read the files while skipping potential comment lines
    df1 = pd.read_csv(file1, delim_whitespace=True, comment='%', header=0)
    df2 = pd.read_csv(file2, delim_whitespace=True, comment='%', header=0)

    # Ensure both files have the same columns
    if list(df1.columns) != list(df2.columns):
        raise ValueError("Columns in both files do not match")

    # Subtract column by column
    result_df = df1 - df2

    # Save the result to a new file with column names and proper formatting
    result_df.to_csv(output_file, sep='\t', index=False, float_format='%.6e')

    print(f"Subtracted data saved to {output_file}")


# Example usage
file1 = "piston_without_energy_equation.txt"  # Replace with actual file path
file2 = "piston.txt"  # Replace with actual file path
output_file = "output.txt"

subtract_txt_files(file1, file2, output_file)