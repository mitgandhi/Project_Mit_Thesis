import vtk

# Define the file path (modify to an actual existing file in your dataset)
vtk_3_sample_file = "G:/Inline/Highspeed_pump/HP_VD/SIM/RUN_1_Straight_cordinate_piston/simulation/output/piston/vtk/piston_gap.200.vtk"

# Load the VTK file
reader = vtk.vtkDataSetReader()  # Auto-detects dataset type
reader.SetFileName(vtk_3_sample_file)
reader.Update()

# Get the output dataset
output_data = reader.GetOutput()

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
