import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


def load_and_visualize(csv_path):
    # Expand user path (handles ~)
    csv_path = os.path.expanduser(csv_path)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Print column names for debugging
    print("Available columns:", df.columns.tolist())

    # Create a new figure with 2 subplots
    fig = plt.figure(figsize=(15, 7))

    # 3D scatter plot of component positions
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(df['center_x'],
                          df['center_y'],
                          df['center_z'],
                          c=df['volume'],  # Color by volume
                          s=100,  # Point size
                          cmap='viridis')

    # Add labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Component Positions')

    # Add colorbar
    plt.colorbar(scatter, label='Volume')

    # Bounding box visualization
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot each component's bounding box
    for _, row in df.iterrows():
        # Get corners of the bounding box
        x = [row['min_x'], row['max_x']]
        y = [row['min_y'], row['max_y']]
        z = [row['min_z'], row['max_z']]

        # Plot bounding box edges
        for i in range(2):
            for j in range(2):
                ax2.plot([x[0], x[1]], [y[i], y[i]], [z[j], z[j]], 'b-', alpha=0.3)
                ax2.plot([x[i], x[i]], [y[0], y[1]], [z[j], z[j]], 'b-', alpha=0.3)
                ax2.plot([x[i], x[i]], [y[j], y[j]], [z[0], z[1]], 'b-', alpha=0.3)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Bounding Boxes')

    # Create a bar chart of component volumes
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    volumes = df.groupby('component_name')['volume'].sum()  # Changed from 'component' to 'component_name'
    volumes.plot(kind='bar', ax=ax3)
    ax3.set_title('Component Volumes')
    ax3.set_xlabel('Component')
    ax3.set_ylabel('Volume')
    plt.xticks(rotation=45, ha='right')  # Added ha='right' for better label alignment

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Use the correct path to your CSV file
    csv_path = "~/Desktop/fusion_measurements.csv"

    try:
        load_and_visualize(csv_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        print(
            "\nIf you're seeing a 'KeyError', please check that the CSV file exists and contains the expected columns:")
        print("- component_name")
        print("- volume")
        print("- center_x, center_y, center_z")
        print("- min_x, min_y, min_z")
        print("- max_x, max_y, max_z")