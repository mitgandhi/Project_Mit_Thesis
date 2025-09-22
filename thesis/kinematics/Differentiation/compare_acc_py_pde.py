import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from py_pde import PDE, ScalarField, CartesianGrid
from py_pde.tools.numerics import central_differences


class MotionAnalysis:
    def __init__(self, R, beta, phi, omega, Bore_length):
        """
        Constructor to initialize motion parameters

        Parameters:
        R (float): Radius
        beta (float): Beta angle in degrees
        phi (float): Phi angle in degrees
        omega (float): Angular velocity in rad/s
        Bore_length (float): Length of bore
        """
        self.R = R
        self.beta = np.deg2rad(beta)
        self.phi = np.deg2rad(phi)
        self.omega = omega
        self.L = Bore_length

    def _calculate_displacement(self, theta):
        """Helper method to calculate displacement using direct formula"""
        K1 = (self.R * np.tan(self.beta)) / np.cos(self.phi)
        K2 = np.tan(self.phi) * np.tan(self.beta)
        return (K1 * (1 - np.cos(theta)) / (1 + K2 * np.cos(theta)))

    def _calculate_velocity(self, theta):
        """Helper method to calculate velocity using direct formula"""
        K1 = (self.R * np.tan(self.beta)) / np.cos(self.phi)
        K2 = np.tan(self.phi) * np.tan(self.beta)
        return K1 * self.omega * (1 + K2) * np.sin(theta) / (1 + K2 * np.cos(theta))

    def _calculate_acc(self, theta):
        """Helper method to calculate acceleration using direct formula"""
        K1 = (self.R * np.tan(self.beta)) / np.cos(self.phi)
        K2 = np.tan(self.phi) * np.tan(self.beta)
        return (
                self.omega ** 2
                * K1
                * (1 + K2)
                * (K2 * (1 + np.sin(theta) * np.sin(theta)) + np.cos(theta))
                / ((1 + K2 * np.cos(theta)) ** 2)
        )

    def method1(self, theta):
        """First method using direct formulas"""
        displacement = self._calculate_displacement(theta)
        velocity = self._calculate_velocity(theta)
        acceleration = self._calculate_acc(theta)
        return displacement, velocity, acceleration

    def method2(self, theta):
        """
        Second method using py_pde for numerical differentiation
        """
        # Create a grid for numerical differentiation
        grid = CartesianGrid([[0, 2 * np.pi]], periodic=True)

        # Calculate displacement
        def displacement_equation(t):
            N = ((self.L - self.R * (np.cos(t) * np.tan(self.beta))) /
                 (np.tan(self.beta) * np.cos(t) * np.sin(self.phi) + np.cos(self.phi)))
            T1 = (self.L / np.cos(self.phi))
            return N - T1

        # Create displacement field
        theta_field = np.linspace(0, 2 * np.pi, len(theta))
        displacement = displacement_equation(theta)

        # Calculate velocity using central differences
        dx = theta_field[1] - theta_field[0]
        velocity = -self.omega * central_differences(displacement, dx, periodic=True)

        # Calculate acceleration using central differences on velocity
        acceleration = self.omega ** 2 * central_differences(velocity, dx, periodic=True)

        # Return values for the specific theta
        idx = np.abs(theta_field - theta).argmin()
        return float(displacement[idx]), float(velocity[idx]), float(acceleration[idx])

    def compare_and_save(self, theta_range, filename='motion_comparison.csv'):
        """Compare both methods and save results to CSV"""
        disp1, vel1, acc1 = [], [], []
        disp2, vel2, acc2 = [], [], []

        for t in theta_range:
            d1, v1, a1 = self.method1(t)
            d2, v2, a2 = self.method2(t)

            disp1.append(d1)
            vel1.append(v1)
            acc1.append(a1)
            disp2.append(d2)
            vel2.append(v2)
            acc2.append(a2)

        df = pd.DataFrame({
            'theta': theta_range,
            'displacement_method1': disp1,
            'velocity_method1': vel1,
            'acceleration_method1': acc1,
            'displacement_method2': disp2,
            'velocity_method2': vel2,
            'acceleration_method2': acc2
        })

        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return df

    def plot_comparison(self, df):
        """Plot comparison of both methods"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot displacement comparison
        ax1.plot(df['theta'], df['displacement_method1'], 'b-', label='Method 1')
        ax1.plot(df['theta'], df['displacement_method2'], 'r--', label='Method 2')
        ax1.set_xlabel('Theta')
        ax1.set_ylabel('Displacement')
        ax1.set_title('Displacement Comparison')
        ax1.grid(True)
        ax1.legend()

        # Plot velocity comparison
        ax2.plot(df['theta'], df['velocity_method1'], 'b-', label='Method 1')
        ax2.plot(df['theta'], df['velocity_method2'], 'r--', label='Method 2')
        ax2.set_xlabel('Theta')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity Comparison')
        ax2.grid(True)
        ax2.legend()

        # Plot acceleration comparison
        ax3.plot(df['theta'], df['acceleration_method1'], 'b-', label='Method 1')
        ax3.plot(df['theta'], df['acceleration_method2'], 'r--', label='Method 2')
        ax3.set_xlabel('Theta')
        ax3.set_ylabel('Acceleration')
        ax3.set_title('Acceleration Comparison')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set parameters
    revolutions = 2
    rpm = 2000
    beta = 15
    phi = 3
    R = 43.5
    Bore_length = 33.25
    omega = 2 * np.pi * rpm / 60

    # Create theta range
    theta = np.linspace(0, 2 * np.pi * revolutions, 360)

    # Create analyzer and get results
    analyzer = MotionAnalysis(R, beta, phi, omega, Bore_length)
    results = analyzer.compare_and_save(theta)

    # Plot results
    analyzer.plot_comparison(results)