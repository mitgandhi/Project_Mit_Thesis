import re


def convert_to_fixed_point(value):
    # Adjust this format for desired precision
    return "{:.13f}".format(float(value))


# Open the original .cdb file and a new file to save converted data
with open("mesh.cdb", "r") as infile, open("mesh_fixed.cdb", "w") as outfile:
    for line in infile:
        # Replace numbers in scientific notation with fixed-point
        modified_line = re.sub(r'([-+]?\d*\.\d+|[-+]?\d+\.?\d*)[eE]([-+]?\d+)',
                               lambda x: convert_to_fixed_point(x.group()), line)
        outfile.write(modified_line)