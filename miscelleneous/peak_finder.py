import pandas as pd
import numpy as np

Peak_to_peak_o = []


class DataFileReader:
    def read_data_file(self, filename):
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
            df = df.dropna(axis=1, how='all')  # Remove empty columns
            df = df.dropna(how='all')  # Remove empty rows
            return df
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return None

    def find_peak_values(self, df, column_name):
        try:
            if column_name not in df.columns:
                print(f"Column '{column_name}' not found in the DataFrame")
                return None
            column = df[column_name]
            max_value = column.max()
            min_value = column.min()
            peak_to_peak = max_value - min_value
            mean_value = column.mean()
            median_value = column.median()
            print(f"Peak Values for Column '{column_name}':")
            print(f"Maximum Value: {max_value}")
            print(f"Minimum Value: {min_value}")
            print(f"Peak-to-Peak Value: {peak_to_peak}")
            print(f"Mean Value: {mean_value}")
            print(f"Median Value: {median_value}")
            Peak_to_peak_o.append(peak_to_peak)
            return (max_value, min_value, peak_to_peak, mean_value, median_value)
        except Exception as e:
            print(f"Error finding peak values: {str(e)}")
            return None


def main():
    reader = DataFileReader()
    filename = 'Z:/Studenten/Mit/Inline_Thesis-Simulation/V60N_inclined_pump/Run/Run2_Test_inclined-V60N/Run_inclined_V60N_inclined-code_Method_4/V60N_S4_HP_dp_350_n_2500_d_100/output/piston/piston.txt'
    filename_1 = 'Z:/Studenten/Mit/Inline_Thesis-Simulation/V60N_inclined_pump/Run/Run1_Test_V60N/Run_V60N/V60N_S4_dp_350_n_2500_d_100/output/piston/piston.txt'

    df = reader.read_data_file(filename)
    df_1 = reader.read_data_file(filename_1)

    if df is not None:
        print("Available Columns for First File:")
        print(df.columns.tolist())
        column_name = 'Stroke'
        reader.find_peak_values(df, column_name)

    if df_1 is not None:
        print("Available Columns for Second File:")
        print(df_1.columns.tolist())
        column_name = 'Stroke'
        reader.find_peak_values(df_1, column_name)

    if len(Peak_to_peak_o) >= 2:
        # Convert to float and calculate percentage
        per = float((Peak_to_peak_o[1] - Peak_to_peak_o[0]) / Peak_to_peak_o[1] * 100)
        print("percentage:" + str(per))
    else:
        print("Not enough peak-to-peak values to calculate percentage difference")


if __name__ == "__main__":
    main()