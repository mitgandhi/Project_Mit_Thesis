import os


def collect_cpp_code(folder_path, output_file):
    # Open output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through the directory
        for root, dirs, files in os.walk(folder_path):
            # Filter for .cpp files
            cpp_files = [f for f in files if f.endswith(('.h'))]

            for file in cpp_files:
                file_path = os.path.join(root, file)
                # Write file name as separator

                outfile.write(f"File: {file_path}\n")


                # Copy contents of the file
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    outfile.write('\n\n')
                except Exception as e:
                    outfile.write(f"Error reading file: {str(e)}\n\n")


# Usage example
folder_path = "E:/Job_1/Influgen_3/Influgen_3/ConsoleApplication1/src/include"  # Replace with your folder path
output_file = "include.txt"  # Output file name
collect_cpp_code(folder_path, output_file)