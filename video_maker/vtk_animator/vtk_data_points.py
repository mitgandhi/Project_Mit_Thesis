import vtk
import os

# Define the file path
vtk_3_sample_file = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run1_Test_V60N\Run_V60N\V60N_S4_dp_350_n_4500_d_100\output\piston\vtk\piston_gap.1800.vtk"

# Check if file exists
if not os.path.exists(vtk_3_sample_file):
    print(f"File does not exist: {vtk_3_sample_file}")
    print("Please check the file path and make sure the file exists.")
    exit()

# Load the VTK file
reader = vtk.vtkDataSetReader()
reader.SetFileName(vtk_3_sample_file)
reader.Update()

# Check if reading was successful
output_data = reader.GetOutput()
if output_data is None:
    print("Failed to read the VTK file.")
    exit()

# Get Cell Data
cell_data = output_data.GetCellData()
print("=== Available CELL Data Arrays ===")
for i in range(cell_data.GetNumberOfArrays()):
    print(f"  - {cell_data.GetArrayName(i)}")

# Get Point Data
point_data = output_data.GetPointData()
print("\n=== Available POINT Data Arrays ===")
for i in range(point_data.GetNumberOfArrays()):
    print(f"  - {point_data.GetArrayName(i)}")