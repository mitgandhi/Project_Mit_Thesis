import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os


class MeshVisualizer:
    def __init__(self, mesh_data):
        self.mesh_data = mesh_data
        self.N, self.M, self.Q = mesh_data['dimensions']
        self.output_dir = 'output/figures'

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_mesh_structure(self, save=False):
        """Plot basic mesh structure"""
        mid_Q = self.Q // 2

        fig = plt.figure(figsize=(15, 5))

        # Plot 1: Top view (XY plane)
        ax1 = fig.add_subplot(131)
        ax1.scatter(self.mesh_data['X'][:, :, mid_Q],
                    self.mesh_data['Y'][:, :, mid_Q],
                    c=self.mesh_data['R'][:, :, mid_Q],
                    cmap='viridis', s=1)
        ax1.set_title('Top View (XY Plane)')
        ax1.set_aspect('equal')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')

        # Plot 2: Side view (XZ plane)
        ax2 = fig.add_subplot(132)
        ax2.scatter(self.mesh_data['X'][:, self.M // 2, :],
                    self.mesh_data['Z'][:, self.M // 2, :],
                    c=self.mesh_data['R'][:, self.M // 2, :],
                    cmap='viridis', s=1)
        ax2.set_title('Side View (XZ Plane)')
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Z [m]')

        # Plot 3: 3D view
        ax3 = fig.add_subplot(133, projection='3d')
        scatter = ax3.scatter(self.mesh_data['X'][:, :, mid_Q].flatten(),
                              self.mesh_data['Y'][:, :, mid_Q].flatten(),
                              self.mesh_data['Z'][:, :, mid_Q].flatten(),
                              c=self.mesh_data['R'][:, :, mid_Q].flatten(),
                              cmap='viridis', s=1)
        ax3.set_title('3D View')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_zlabel('Z [m]')

        plt.colorbar(scatter, ax=ax3, label='Radius [m]')
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, 'mesh_structure.png'), dpi=300)
        plt.show()

    def plot_field_distribution(self, field, field_name, units, save=False):
        """Plot distribution of a given field"""
        mid_Q = self.Q // 2

        fig = plt.figure(figsize=(15, 5))

        # Plot 1: Contour in XY plane
        ax1 = fig.add_subplot(131)
        im1 = ax1.contourf(self.mesh_data['X'][:, :, mid_Q],
                           self.mesh_data['Y'][:, :, mid_Q],
                           field[:, :, mid_Q], levels=20)
        ax1.set_title(f'{field_name} (XY Plane)')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label=f'{field_name} [{units}]')

        # Plot 2: Angular distribution
        ax2 = fig.add_subplot(132, projection='polar')
        theta = self.mesh_data['Theta'][:, :, mid_Q]
        r = self.mesh_data['R'][:, :, mid_Q]
        im2 = ax2.contourf(theta, r, field[:, :, mid_Q])
        ax2.set_title(f'{field_name} (Polar View)')
        plt.colorbar(im2, ax=ax2, label=f'{field_name} [{units}]')

        # Plot 3: 3D surface
        ax3 = fig.add_subplot(133, projection='3d')
        surf = ax3.plot_surface(self.mesh_data['X'][:, :, mid_Q],
                                self.mesh_data['Y'][:, :, mid_Q],
                                field[:, :, mid_Q],
                                cmap='viridis')
        ax3.set_title(f'{field_name} (3D View)')
        plt.colorbar(surf, ax=ax3, label=f'{field_name} [{units}]')

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(self.output_dir, f'{field_name.lower()}_distribution.png'), dpi=300)
        plt.show()

    def save_results_summary(self, forces, filename='forces_summary.txt'):
        """Save numerical results to file"""
        filepath = os.path.join('output/results', filename)
        os.makedirs('output/results', exist_ok=True)

        with open(filepath, 'w') as f:
            f.write("=== Force Calculation Results ===\n\n")

            f.write("Fluid Forces:\n")
            f.write(f"FfK = {forces.FfK:.2e} N\n")
            f.write(f"FfKx = {forces.FfKx:.2e} N\n")
            f.write(f"FfKy = {forces.FfKy:.2e} N\n")
            f.write(f"MfKx = {forces.MfKx:.2e} N⋅m\n")
            f.write(f"MfKy = {forces.MfKy:.2e} N⋅m\n\n")

            f.write("Contact Forces:\n")
            f.write(f"FcK = {forces.FcK:.2e} N\n")
            f.write(f"FcKx = {forces.FcKx:.2e} N\n")
            f.write(f"FcKy = {forces.FcKy:.2e} N\n")
            f.write(f"McKx = {forces.McKx:.2e} N⋅m\n")
            f.write(f"McKy = {forces.McKy:.2e} N⋅m\n\n")

            f.write("Force Differences:\n")
            for i, force in enumerate(forces.dF):
                f.write(f"dF[{i}] = {force:.2e} N\n")