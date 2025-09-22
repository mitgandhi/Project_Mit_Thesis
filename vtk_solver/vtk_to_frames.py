import vtk
import math
import os
import subprocess
from pathlib import Path
import time
import shutil


class AnimationController:
    def __init__(self, render_window, callback_function, update_vtk_3):
        self.render_window = render_window
        self.callback_function = callback_function
        self.update_vtk_3 = update_vtk_3  # Function to update VTK 3 set
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 360  # FULL 360-degree revolution
        self.timer_id = None

    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        print("Animation playing" if self.is_playing else "Animation paused")

    def timer_callback(self, obj, event):
        if self.is_playing:
            self.current_frame = (self.current_frame + 1) % self.total_frames
            angle = 360 - self.current_frame  # CLOCKWISE rotation (changed from self.current_frame)
            self.callback_function(angle)
            self.update_vtk_3(angle)
            self.render_window.Render()

    def export_video(self):
        print("Starting video export...")

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.SetInputBufferTypeToRGBA()  # Changed from RGB to RGBA for transparency
        window_to_image_filter.ReadFrontBufferOff()

        temp_dir = Path("./temp_frames")
        temp_dir.mkdir(exist_ok=True)

        was_playing = self.is_playing
        self.is_playing = False

        # Calculate fewer frames for 10-second video (30fps x 10s = 300 frames)
        frames_for_video = 300

        for frame_idx in range(frames_for_video):
            # Map current frame index to full revolution (0-359 degrees)
            frame = int((frame_idx / frames_for_video) * self.total_frames)
            angle = 360 - frame  # CLOCKWISE rotation (changed from frame)
            self.callback_function(angle)
            self.update_vtk_3(angle)
            self.render_window.Render()

            window_to_image_filter.Modified()
            window_to_image_filter.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(f"./temp_frames/frame_{frame_idx:04d}.png")
            writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            writer.Write()

            print(f"Frame {frame_idx + 1}/{frames_for_video} saved")

        # Try to find ffmpeg automatically
        ffmpeg_cmd = self.find_ffmpeg()

        if ffmpeg_cmd:
            print(f"Using FFmpeg: {ffmpeg_cmd}")
            try:
                # Create MP4 video
                subprocess.run([
                    ffmpeg_cmd,
                    "-framerate", "30",
                    "-i", "./temp_frames/frame_%04d.png",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-y", "animation.mp4"
                ], check=True)

                # Create MOV with transparency
                subprocess.run([
                    ffmpeg_cmd,
                    "-framerate", "30",
                    "-i", "./temp_frames/frame_%04d.png",
                    "-c:v", "qtrle",
                    "-pix_fmt", "rgba",
                    "-y", "animation_transparent.mov"
                ], check=True)

                print("Videos exported to animation.mp4 and animation_transparent.mov (for PowerPoint)")
            except subprocess.CalledProcessError as e:
                print(f"Error running FFmpeg: {e}")
        else:
            print("FFmpeg not found. Please install FFmpeg or specify the path manually.")
            print("PNG frames have been saved to the 'temp_frames' folder.")

        if was_playing:
            self.is_playing = True

    def find_ffmpeg(self):
        """Try to find ffmpeg in common locations or PATH"""
        # First check if ffmpeg is in PATH
        if shutil.which("ffmpeg"):
            return "ffmpeg"

        # Common locations for Windows
        windows_paths = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Users\{}\AppData\Local\Programs\ffmpeg\bin\ffmpeg.exe".format(os.getlogin())
        ]

        # Common locations for Linux/Mac
        unix_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg"
        ]

        # Check all possible paths
        possible_paths = windows_paths if os.name == 'nt' else unix_paths

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None


def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {os.path.basename(filepath)}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False


def load_vtk_file_safe(reader, filepath, description):
    """Safely load a VTK file with error checking"""
    if not check_file_exists(filepath, description):
        return False

    try:
        reader.SetFileName(filepath)
        reader.Update()

        # Check if data was loaded
        output = reader.GetOutput()
        if output is None:
            print(f"✗ Failed to read {description}")
            return False
        elif hasattr(output, 'GetNumberOfPoints') and output.GetNumberOfPoints() == 0:
            print(f"✗ {description} contains no data points")
            return False
        else:
            print(f"✓ Successfully loaded {description}")
            return True
    except Exception as e:
        print(f"✗ Error loading {description}: {e}")
        return False


# Define file paths
base_path = r"Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run1_Test_V60N\Run_V60N\V60N_S4_dp_350_n_4500_d_100\output\piston\vtk"
print(f"Base directory: {base_path}")

# Check if base directory exists
if not os.path.exists(base_path):
    print(f"✗ Base directory does not exist: {base_path}")
    print("Please check the path and make sure it's accessible")
    exit(1)
else:
    print(f"✓ Base directory exists")

# List some files in the directory for debugging
try:
    files = os.listdir(base_path)
    vtk_files = [f for f in files if f.endswith('.vtk')]
    print(f"Found {len(vtk_files)} VTK files in directory")
    if vtk_files:
        print("Sample files:")
        for i, f in enumerate(vtk_files[:5]):  # Show first 5 files
            print(f"  - {f}")
        if len(vtk_files) > 5:
            print(f"  ... and {len(vtk_files) - 5} more files")
except Exception as e:
    print(f"Error listing directory contents: {e}")

# Define specific file paths
cylinder_file = os.path.join(base_path, "cylinder_th_rev_5.vtk")
piston_file = os.path.join(base_path, "piston_th_rev_5.vtk")
gap_sample_file = os.path.join(base_path, "piston_gap.2159.vtk")  # Test with frame 2159

print("\nChecking required files...")

# Create common lookup table for temperature (VTK 1 and VTK 2)
temp_lut = vtk.vtkLookupTable()
temp_lut.SetTableRange(40, 60)  # Temperature range from 40°C to 60°C
temp_lut.SetNumberOfColors(256)  # High resolution for smooth gradient
temp_lut.SetRampToLinear()  # Linear color interpolation

# Manually define the color transition from Red -> Gray -> Blue
color_transfer_function = vtk.vtkColorTransferFunction()
color_transfer_function.AddRGBPoint(40, 1.0, 0.0, 0.0)  # Red at 40°C
color_transfer_function.AddRGBPoint(50, 0.5, 0.5, 0.5)  # Gray at 50°C
color_transfer_function.AddRGBPoint(60, 0.0, 0.0, 1.0)  # Blue at 60°C

# Apply the transfer function to the lookup table
for i in range(256):
    t = 40 + (i / 255) * 20  # Normalize range from 40°C to 60°C
    r, g, b = color_transfer_function.GetColor(t)
    temp_lut.SetTableValue(i, r, g, b, 1.0)  # RGB + Alpha = 1.0 (Opaque)

temp_lut.Build()

# Create lookup table for pressure (VTK 3)
pressure_lut = vtk.vtkLookupTable()
pressure_lut.SetTableRange(0, 100)  # Pressure range from 0 to 100 bar
pressure_lut.SetHueRange(0.667, 0.0)  # Blue to red
pressure_lut.SetNumberOfColors(256)
pressure_lut.Build()

# ====================== VTK 1 (Cylinder) ======================
print("\nLoading VTK 1 (Cylinder)...")
reader1 = vtk.vtkUnstructuredGridReader()
reader1.ReadAllScalarsOn()
reader1.ReadAllFieldsOn()

if not load_vtk_file_safe(reader1, cylinder_file, "Cylinder file"):
    print("Cannot continue without cylinder file")
    exit(1)

# Convert cell data to point data for smoother coloring - this is key for smooth visualization
cell_to_point1 = vtk.vtkCellDataToPointData()
cell_to_point1.SetInputConnection(reader1.GetOutputPort())
cell_to_point1.PassCellDataOn()  # Also keep the original cell data
cell_to_point1.Update()

# Use geometry filter to get surface
geometry_filter1 = vtk.vtkDataSetSurfaceFilter()
geometry_filter1.SetInputConnection(cell_to_point1.GetOutputPort())
geometry_filter1.Update()

# Define cutting plane
plane = vtk.vtkPlane()
plane.SetOrigin(0, 0, 0)
plane.SetNormal(1, 0, 0)

# Clip the data
clipper1 = vtk.vtkClipPolyData()
clipper1.SetInputConnection(geometry_filter1.GetOutputPort())
clipper1.SetClipFunction(plane)
clipper1.SetGenerateClippedOutput(True)
clipper1.Update()

# Create mapper
mapper1 = vtk.vtkPolyDataMapper()
mapper1.SetInputConnection(clipper1.GetOutputPort())
mapper1.SetScalarModeToUsePointData()  # Use point data for smoother interpolation
mapper1.SelectColorArray("T_[C]")
mapper1.SetLookupTable(temp_lut)
mapper1.SetScalarRange(40, 60)
mapper1.SetScalarVisibility(True)
mapper1.InterpolateScalarsBeforeMappingOn()  # For smoother color interpolation

# Create actor
actor1 = vtk.vtkActor()
actor1.SetMapper(mapper1)
actor1.GetProperty().SetInterpolationToPhong()  # Smooth shading

# ====================== VTK 2 (Piston) ======================
print("\nLoading VTK 2 (Piston)...")
reader2 = vtk.vtkUnstructuredGridReader()
reader2.ReadAllScalarsOn()
reader2.ReadAllFieldsOn()

if not load_vtk_file_safe(reader2, piston_file, "Piston file"):
    print("Cannot continue without piston file")
    exit(1)

# Convert cell data to point data for smoother coloring
cell_to_point2 = vtk.vtkCellDataToPointData()
cell_to_point2.SetInputConnection(reader2.GetOutputPort())
cell_to_point2.PassCellDataOn()  # Also keep the original cell data
cell_to_point2.Update()

# Use geometry filter to get surface
geometry_filter2 = vtk.vtkDataSetSurfaceFilter()
geometry_filter2.SetInputConnection(cell_to_point2.GetOutputPort())
geometry_filter2.Update()

# Create mapper
mapper2 = vtk.vtkPolyDataMapper()
mapper2.SetInputConnection(geometry_filter2.GetOutputPort())
mapper2.SetScalarModeToUsePointData()  # Use point data for smoother interpolation
mapper2.SelectColorArray("T_[C]")
mapper2.SetLookupTable(temp_lut)
mapper2.SetScalarRange(40, 60)
mapper2.SetScalarVisibility(True)
mapper2.InterpolateScalarsBeforeMappingOn()  # For smoother color interpolation

# Create actor
actor2 = vtk.vtkActor()
actor2.SetMapper(mapper2)
actor2.GetProperty().SetOpacity(0.3)
actor2.GetProperty().SetInterpolationToPhong()  # Smooth shading

# ====================== VTK 3 (Gap Pressure) ======================
print("\nLoading VTK 3 (Gap Pressure)...")
reader3 = vtk.vtkDataSetReader()
reader3.ReadAllScalarsOn()
reader3.ReadAllFieldsOn()

# Test with a sample gap file first
if not load_vtk_file_safe(reader3, gap_sample_file, "Gap pressure sample file"):
    print("Warning: Cannot load gap pressure files. Animation will work without pressure visualization.")

# Create mapper
mapper3 = vtk.vtkPolyDataMapper()
mapper3.SetScalarModeToUsePointData()  # Already using point data
mapper3.SelectColorArray("Gap_Pressure_[bar]")
mapper3.SetLookupTable(pressure_lut)
mapper3.SetScalarRange(0, 100)
mapper3.SetScalarVisibility(True)
mapper3.InterpolateScalarsBeforeMappingOn()  # For smoother color interpolation

# Create actor
actor3 = vtk.vtkActor()
actor3.SetMapper(mapper3)
actor3.GetProperty().SetInterpolationToPhong()  # Smooth shading


# Function to update the third VTK file based on angle
def update_vtk_3(angle):
    try:
        # Map angle (0-359) to range 1800-2159
        vtk_3_filename = os.path.join(base_path, f"piston_gap.{2159 - angle}.vtk")

        if os.path.exists(vtk_3_filename):
            reader3.SetFileName(vtk_3_filename)
            reader3.Update()

            # Use surface filter to get geometry
            geometry_filter3 = vtk.vtkDataSetSurfaceFilter()
            geometry_filter3.SetInputConnection(reader3.GetOutputPort())
            geometry_filter3.Update()

            mapper3.SetInputConnection(geometry_filter3.GetOutputPort())

            # Apply transformation (adjust if necessary)
            transform3 = vtk.vtkTransform()
            transform3.Identity()

            # transform3.RotateZ(360 - angle)  # Remove rotation
            transform3.Translate(0, 0.04352, 0)
            # transform3.RotateX(-5)

            actor3.SetUserTransform(transform3)
        else:
            # print(f"Warning: Gap file not found for angle {angle}: {vtk_3_filename}")
            pass  # Silent skip for missing files
    except Exception as e:
        print(f"Error updating VTK 3 for angle {angle}: {e}")


# Add color bar actors for temperature and pressure
# Temperature scalar bar
temp_scalar_bar = vtk.vtkScalarBarActor()
temp_scalar_bar.SetOrientationToHorizontal()
temp_scalar_bar.SetLookupTable(temp_lut)
temp_scalar_bar.SetTitle("Temperature [°C]")
temp_scalar_bar.SetNumberOfLabels(5)
temp_scalar_bar.SetPosition(0.05, 0.05)
temp_scalar_bar.SetWidth(0.4)
temp_scalar_bar.SetHeight(0.05)
temp_scalar_bar.SetLabelFormat("%.1f")  # Set precision of labels

# Pressure scalar bar
pressure_scalar_bar = vtk.vtkScalarBarActor()
pressure_scalar_bar.SetOrientationToHorizontal()
pressure_scalar_bar.SetLookupTable(pressure_lut)
pressure_scalar_bar.SetTitle("Gap Pressure [bar]")
pressure_scalar_bar.SetNumberOfLabels(5)
pressure_scalar_bar.SetPosition(0.5, 0.05)
pressure_scalar_bar.SetWidth(0.4)
pressure_scalar_bar.SetHeight(0.05)
pressure_scalar_bar.SetLabelFormat("%.1f")  # Set precision of labels

# Add actors and configure rendering
print("\nSetting up renderer...")
renderer = vtk.vtkRenderer()
# Changed from black (0, 0, 0) to white (1, 1, 1) background
renderer.SetBackground(1, 1, 1)  # White background

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 800)
render_window.SetWindowName("Pump Animation with Controls")
render_window.SetAlphaBitPlanes(1)  # Enable alpha channel for transparency
render_window.SetMultiSamples(0)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

renderer.AddActor(actor1)
renderer.AddActor(actor2)
renderer.AddActor(actor3)  # Add third VTK file
renderer.AddViewProp(temp_scalar_bar)  # Use AddViewProp instead of deprecated AddActor2D
renderer.AddViewProp(pressure_scalar_bar)  # Use AddViewProp instead of deprecated AddActor2D
renderer.UseHiddenLineRemovalOn()

bounds1 = actor1.GetBounds()
bounds2 = actor2.GetBounds()
center = [
    (bounds1[0] + bounds1[1] + bounds2[0] + bounds2[1]) / 4,
    (bounds1[2] + bounds1[3] + bounds2[2] + bounds2[3]) / 4,
    (bounds1[4] + bounds1[5] + bounds2[4] + bounds2[5]) / 4,
]

camera = renderer.GetActiveCamera()

# Move the camera far along the X-axis, looking towards the YZ plane
camera.SetPosition(center[0] - 10, center[1], center[2])  # Move camera to -X
camera.SetFocalPoint(center[0], center[1], center[2])  # Focus on the center
camera.SetViewUp(0, 0, 1)  # Keep Z as up

# Force orthographic projection (makes it 2D)
camera.ParallelProjectionOn()

# Reset camera to apply changes
renderer.ResetCamera()
renderer.ResetCameraClippingRange()


def animate_models(angle):
    # Calculate displacement for piston movement
    displacement = (angle / 180.0) * 0.02 if angle <= 180 else ((360 - angle) / 180.0) * 0.02

    # Transform for VTK 2 (piston)
    transform = vtk.vtkTransform()
    transform.Identity()
    # transform.RotateZ(360 - angle)  # CLOCKWISE rotation (changed from angle)
    transform.Translate(0, 0.04395, -displacement + 0.019)
    # transform.RotateX(-5)

    actor2.SetUserTransform(transform)


# Create animation controller and set up interaction
print("Setting up animation controller...")
controller = AnimationController(render_window, animate_models, update_vtk_3)
render_window_interactor.AddObserver(vtk.vtkCommand.TimerEvent, controller.timer_callback)

# Initialize visualizer
print("Initializing VTK window...")
render_window_interactor.Initialize()
render_window_interactor.CreateRepeatingTimer(50)

# Update VTK 3 with initial angle
print("Loading initial frame...")
update_vtk_3(0)

# Initial render
render_window.Render()

print("\n" + "=" * 50)
print("VTK Animation Ready!")
print("=" * 50)
print("Controls:")
print("- Press 'Space' to toggle play/pause")
print("- Press 'E' to export animation as MP4 (10 seconds) and MOV (with transparency)")
print("- Close the window or press Ctrl+C to exit")
print("=" * 50)


# Set up key bindings
def key_press_event(obj, event):
    key = obj.GetKeySym().lower()
    if key == "e":
        controller.export_video()
    elif key == "space":
        controller.toggle_play_pause()
    elif key == "escape" or key == "q":
        print("Exiting...")
        render_window_interactor.ExitCallback()


render_window_interactor.AddObserver("KeyPressEvent", key_press_event)

# Start the interactor
print("Starting VTK interactor...")
try:
    render_window_interactor.Start()
except KeyboardInterrupt:
    print("\nExiting...")
except Exception as e:
    print(f"Error during interaction: {e}")

print("VTK session ended.")