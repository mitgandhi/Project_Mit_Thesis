import pandas as pd
# Display the new CSV file to the user

# Load the CSV file
file_path = "P_DC.csv"
df = pd.read_csv(file_path)

# Select every fifth value from the dataframe
df_subset = df.iloc[::5]
# Convert the values to double (float type)
df_subset = df_subset.astype(int)



# Save the subset to a new CSV file
new_file_path = "../Results/P_DC_subset.csv"
df_subset.to_csv(new_file_path, index=False)



