import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import derivative


class MotionAnalysis:
    def __init__(self, R, beta, phi, omega, Bore_length):
        """
        Constructor to initialize motion parameters

        Parameters:
        R (float): Radius
        beta (float): Beta angle in degrees
        phi (float): Phi angle in degrees
        omega (float): Angular velocity in rad/s
        Bore_length (float): Length of the bore
        """
        self.R = R
        self.beta = np.deg2rad(beta)
        self.phi = np.deg2rad(phi)
        self.omega = omega
        self.L = Bore_length

    def method1(self, theta):
        """
        First method using direct analytical equations
        """
        K1 = (self.R * np.tan(self.beta)) / np.cos(self.phi)
        K2 = np.tan(self.phi) * np.tan(self.beta)

        # Displacement
        displacement = K1 * (1 - np.cos(theta) / (1 + K2 * np.cos(theta)))

        # Velocity
        velocity = K1 * self.omega * (1 + K2) * np.sin(theta) / (1 + K2 * np.cos(theta))

        # Acceleration
        acceleration = (self.omega ** 2 * K1 * (1 + K2) *
                        (K2 * (1 + np.sin(theta) ** 2) + np.cos(theta)) /
                        ((1 + K2 * np.cos(theta)) ** 2))

        return displacement, velocity, acceleration

    def method2(self, theta):
        """
        Second method using geometric transformation and numerical differentiation
        """

        def displacement_equation(t):
            # Geometric transformation based displacement
            N = ((self.L - self.R * (np.cos(t) * np.tan(self.beta))) /
                 (np.tan(self.beta) * np.cos(t) * np.sin(self.phi) + np.cos(self.phi)))
            T1 = self.L / np.cos(self.phi)
            return N - T1

        # Calculate displacement directly
        displacement = displacement_equation(theta)

        # Calculate velocity using central difference
        def velocity_numerical(t):
            h = 1e-7  # Small step size for numerical differentiation
            return derivative(displacement_equation, t, dx=h)

        # Calculate acceleration using central difference
        def acceleration_numerical(t):
            h = 1e-7  # Small step size for numerical differentiation
            return derivative(lambda x: velocity_numerical(x), t, dx=h)

        velocity = -self.omega * velocity_numerical(theta)
        acceleration = self.omega ** 2 * acceleration_numerical(theta)

        return displacement, velocity, acceleration

    def method3(self, theta):
        """
        Third method using simplified geometric approach with finite differences
        """
        dt = 1e-6  # Time step for finite differences

        # Basic displacement calculation
        def displacement_simplified(t):
            return (self.R * np.tan(self.beta) * (1 - np.cos(t)) /
                    (np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.beta) * np.cos(t))))

        # Calculate displacement
        displacement = displacement_simplified(theta)

        # Calculate velocity using finite differences
        velocity = (displacement_simplified(theta + dt) - displacement_simplified(theta - dt)) / (2 * dt)
        velocity *= self.omega

        # Calculate acceleration using finite differences
        acc_fwd = (displacement_simplified(theta + dt) - 2 * displacement_simplified(theta) +
                   displacement_simplified(theta - dt)) / (dt ** 2)
        acceleration = acc_fwd * self.omega ** 2

        return displacement, velocity, acceleration

    def compare_and_save(self, theta_range, filename='motion_comparison.csv'):
        """
        Compare all three methods and save results to CSV
        """
        results = {
            'theta': theta_range,
            'displacement_method1': [],
            'velocity_method1': [],
            'acceleration_method1': [],
            'displacement_method2': [],
            'velocity_method2': [],
            'acceleration_method2': [],
            'displacement_method3': [],
            'velocity_method3': [],
            'acceleration_method3': []
        }

        for t in theta_range:
            # Method 1
            d1, v1, a1 = self.method1(t)
            results['displacement_method1'].append(d1)
            results['velocity_method1'].append(v1)
            results['acceleration_method1'].append(a1)

            # Method 2
            d2, v2, a2 = self.method2(t)
            results['displacement_method2'].append(d2)
            results['velocity_method2'].append(v2)
            results['acceleration_method2'].append(a2)

            # Method 3
            d3, v3, a3 = self.method3(t)
            results['displacement_method3'].append(d3)
            results['velocity_method3'].append(v3)
            results['acceleration_method3'].append(a3)

        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return df

    def plot_comparison(self, df):
        """
        Plot comparison of all three methods
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Plot styles
        styles = [('Method 1', 'b-'), ('Method 2', 'r--'), ('Method 3', 'g:')]

        # Plot displacement comparison
        for method, style in styles:
            ax1.plot(df['theta'], df[f'displacement_{method.lower().replace(" ", "")}'],
                     style, label=method)
        ax1.set_xlabel('Theta')
        ax1.set_ylabel('Displacement')
        ax1.set_title('Displacement Comparison')
        ax1.grid(True)
        ax1.legend()

        # Plot velocity comparison
        for method, style in styles:
            ax2.plot(df['theta'], df[f'velocity_{method.lower().replace(" ", "")}'],
                     style, label=method)
        ax2.set_xlabel('Theta')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity Comparison')
        ax2.grid(True)
        ax2.legend()

        # Plot acceleration comparison
        for method, style in styles:
            ax3.plot(df['theta'], df[f'acceleration_{method.lower().replace(" ", "")}'],
                     style, label=method)
        ax3.set_xlabel('Theta')
        ax3.set_ylabel('Acceleration')
        ax3.set_title('Acceleration Comparison')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Test parameters
    revolutions = 6
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

    # Plot and save results
    fig = analyzer.plot_comparison(results)
    plt.savefig('motion_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()