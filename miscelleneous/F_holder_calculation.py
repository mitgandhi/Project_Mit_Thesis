import pandas as pd
import matplotlib.pyplot as plt


def read_data_file(filename):
    try:
        # First try to read as standard CSV
        df = pd.read_csv(filename)
        # If that fails, try reading with different delimiters
        if len(df.columns) == 1:
            for delimiter in ['\t', ' ', ';']:
                try:
                    df = pd.read_csv(filename, delimiter=delimiter)
                    if len(df.columns) > 1:
                        break
                except:
                    continue
        # Clean the data
        df = df.dropna(axis=1, how='all')  # Remove empty columns
        df = df.dropna(how='all')  # Remove empty rows
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def main():
    filename = "Z:/Studenten/Mit/Sunfab_test/1000rpm/0/output/slipper/slipper.txt"
    df = read_data_file(filename)

    if df is not None:
        # Extract the three F_holder columns
        holder_columns = ['F_holder1', 'F_holder2', 'F_holder3']

        # Check if all columns exist in the dataframe
        missing_columns = [col for col in holder_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: The following columns are missing: {missing_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            return

        # Create a new dataframe with just the F_holder columns
        holder_df = df[holder_columns].copy()

        # Add a new column that is the sum of the three F_holder columns
        holder_df['F_holder_sum'] = holder_df.sum(axis=1)

        # Plot the sum against the rev column
        plt.figure(figsize=(10, 6))
        plt.plot(df['rev'], holder_df['F_holder_sum'], 'b-', linewidth=2)
        plt.xlabel('Revolution')
        plt.ylabel('Sum of F_holder Values')
        plt.title('Total F_holder vs Revolution')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig('F_holder_sum_plot.png')
        plt.show()

        # Print the first few rows of the new dataframe
        print("\nNew DataFrame with F_holder columns and sum:")
        print(holder_df.head())

        # Optionally save the new dataframe to a CSV file
        holder_df.to_csv('F_holder_summary.csv', index=False)
        print("\nNew DataFrame saved to 'F_holder_summary.csv'")
    else:
        print("Failed to read the data file.")


if __name__ == '__main__':
    main()