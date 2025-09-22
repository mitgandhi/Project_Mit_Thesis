import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from scipy.spatial import ConvexHull


def read_stl(file_path):
    """Read STL file and extract vertices and faces"""
    vertices = []
    faces = []

    with open(file_path, 'rb') as f:
        header = f.read(80)  # Skip header
        face_count = int.from_bytes(f.read(4), 'little')

        for i in range(face_count):
            # Skip normal vector
            f.read(12)

            # Read vertices
            face_vertices = []
            for j in range(3):
                x = float(np.frombuffer(f.read(4), dtype=np.float32)[0])
                y = float(np.frombuffer(f.read(4), dtype=np.float32)[0])
                z = float(np.frombuffer(f.read(4), dtype=np.float32)[0])
                face_vertices.append([x, y, z])

            # Skip attribute byte count
            f.read(2)

            # Add vertices and face
            face_idx = len(vertices)
            vertices.extend(face_vertices)
            faces.append([face_idx, face_idx + 1, face_idx + 2])

    return np.array(vertices), np.array(faces)


def calculate_dimensions(vertices):
    """Calculate comprehensive dimensions of the model"""
    # Basic dimensions
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)

    dimensions = {
        # Overall dimensions
        'length': x_max - x_min,
        'width': y_max - y_min,
        'height': z_max - z_min,

        # Bounding box coordinates
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'z_min': z_min,
        'z_max': z_max,

        # Center point
        'center_x': (x_max + x_min) / 2,
        'center_y': (y_max + y_min) / 2,
        'center_z': (z_max + z_min) / 2,

        # Volume approximation using convex hull
        'convex_hull_volume': ConvexHull(vertices).volume,

        # Surface area approximation
        'surface_area': calculate_surface_area(vertices),

        # Diagonal measurements
        'diagonal_3d': np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2),
        'diagonal_xy': np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2),
        'diagonal_xz': np.sqrt((x_max - x_min) ** 2 + (z_max - z_min) ** 2),
        'diagonal_yz': np.sqrt((y_max - y_min) ** 2 + (z_max - z_min) ** 2),
    }

    return dimensions


def calculate_surface_area(vertices):
    """Calculate approximate surface area"""
    # Calculate unique triangles
    unique_vertices = np.unique(vertices, axis=0)
    hull = ConvexHull(unique_vertices)
    return hull.area


def save_coordinates_csv(vertices, dimensions, output_path):
    """Save vertices and dimensions to CSV files"""
    # Save vertices
    df_vertices = pd.DataFrame(vertices, columns=['x', 'y', 'z'])
    vertices_path = output_path.replace('.csv', '_vertices.csv')
    df_vertices.to_csv(vertices_path, index=False)

    # Save dimensions
    df_dimensions = pd.DataFrame([dimensions])
    dimensions_path = output_path.replace('.csv', '_dimensions.csv')
    df_dimensions.to_csv(dimensions_path, index=False)

    print(f"Vertices saved to: {vertices_path}")
    print(f"Dimensions saved to: {dimensions_path}")


def visualize_model(vertices, faces, dimensions):
    """Create enhanced 3D visualization of the model"""
    fig = plt.figure(figsize=(15, 10))

    # 3D Model View
    ax1 = fig.add_subplot(121, projection='3d')
    mesh = Poly3DCollection(vertices[faces])
    mesh.set_edgecolor('k')
    mesh.set_facecolor('b')
    mesh.set_alpha(0.1)
    ax1.add_collection3d(mesh)

    # Add bounding box
    plot_bounding_box(ax1, dimensions)

    # Set axis limits
    ax1.set_xlim(dimensions['x_min'], dimensions['x_max'])
    ax1.set_ylim(dimensions['y_min'], dimensions['y_max'])
    ax1.set_zlim(dimensions['z_min'], dimensions['z_max'])

    # Labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Model with Bounding Box')

    # Equal aspect ratio
    ax1.set_box_aspect([1, 1, 1])

    # Dimension Visualization
    ax2 = fig.add_subplot(122)
    visualize_dimensions(ax2, dimensions)

    plt.tight_layout()
    plt.show()


def plot_bounding_box(ax, dims):
    """Plot bounding box with dimensions"""
    # Define the vertices of the bounding box
    x_min, x_max = dims['x_min'], dims['x_max']
    y_min, y_max = dims['y_min'], dims['y_max']
    z_min, z_max = dims['z_min'], dims['z_max']

    # Plot edges
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            ax.plot([x, x], [y, y], [z_min, z_max], 'r--', alpha=0.5)
    for x in [x_min, x_max]:
        for z in [z_min, z_max]:
            ax.plot([x, x], [y_min, y_max], [z, z], 'r--', alpha=0.5)
    for y in [y_min, y_max]:
        for z in [z_min, z_max]:
            ax.plot([x_min, x_max], [y, y], [z, z], 'r--', alpha=0.5)


def visualize_dimensions(ax, dimensions):
    """Create a dimension summary visualization"""
    # Create bar chart of main dimensions
    main_dims = {
        'Length': dimensions['length'],
        'Width': dimensions['width'],
        'Height': dimensions['height'],
        '3D Diagonal': dimensions['diagonal_3d']
    }

    ax.bar(main_dims.keys(), main_dims.values())
    ax.set_title('Main Dimensions')
    ax.set_ylabel('Size (units)')
    plt.xticks(rotation=45)


def analyze_model(vertices, dimensions):
    """Print enhanced model statistics"""
    print("\nModel Statistics:")
    print(f"Number of vertices: {len(vertices)}")
    print("\nMain Dimensions:")
    print(f"Length: {dimensions['length']:.2f}")
    print(f"Width: {dimensions['width']:.2f}")
    print(f"Height: {dimensions['height']:.2f}")
    print(f"3D Diagonal: {dimensions['diagonal_3d']:.2f}")
    print(f"\nVolume (approximation): {dimensions['convex_hull_volume']:.2f}")
    print(f"Surface Area (approximation): {dimensions['surface_area']:.2f}")
    print("\nCenter Point:")
    print(f"X: {dimensions['center_x']:.2f}")
    print(f"Y: {dimensions['center_y']:.2f}")
    print(f"Z: {dimensions['center_z']:.2f}")


if __name__ == "__main__":
    # Path to your STL file
    stl_path = os.path.expanduser("~/Desktop/fusion_stl_export/complete_assembly.stl")
    csv_path = os.path.expanduser("~/Desktop/fusion_stl_export/model_coordinates.csv")

    try:
        # Read STL file
        print("Reading STL file...")
        vertices, faces = read_stl(stl_path)

        # Calculate dimensions
        print("Calculating dimensions...")
        dimensions = calculate_dimensions(vertices)

        # Save data
        save_coordinates_csv(vertices, dimensions, csv_path)

        # Analyze model
        analyze_model(vertices, dimensions)

        # Visualize model
        print("\nCreating visualization...")
        visualize_model(vertices, faces, dimensions)

    except Exception as e:
        print(f"Error: {str(e)}")