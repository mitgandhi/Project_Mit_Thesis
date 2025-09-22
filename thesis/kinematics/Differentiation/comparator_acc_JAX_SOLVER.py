import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad


class MotionAnalysis:
    def __init__(self, R, beta, phi, omega, Bore_length):
        """
        Constructor to initialize motion parameters

        Parameters:
        R (float): Radius
        beta (float): Beta angle in degrees
        phi (float): Phi angle in degrees
        omega (float, optional): Angular velocity in rad/s. If not provided, can be set later.
        """
        self.R = R  # radius
        # Convert angles to radians for calculations
        self.beta = np.deg2rad(beta)  # beta angle in radians
        self.phi = np.deg2rad(phi)  # phi angle in radians
        self.omega = omega  # angular velocity
        self.L = Bore_length


    def _calculate_displacement(self, theta):
        """Helper method to calculate displacement"""
        # K1 = (self.R * (np.tan(self.beta))) / (
        #     np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.beta))
        # )
        K1= (self.R * (np.tan(self.beta))) / (
            np.cos(self.phi))
        K2 = np.tan(self.phi) * np.tan(self.beta)
        return (K1*(1-np.cos(theta)/ (1+ K2* np.cos(theta))))

    def _calcualte_velocity(self, theta):

        # K1 = (self.R * (np.tan(self.beta))) / (
        #     np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.beta))
        # )
        K1 = (self.R * (np.tan(self.beta))) / (
            np.cos(self.phi))
        K2 = np.tan(self.phi) * np.tan(self.beta)
        return K1 * self.omega * (1 + K2) * np.sin(theta) / (1 + K2 * np.cos(theta))

    def _calcualte_acc(self, theta):

        # K1 = (self.R * (np.tan(self.beta))) / (
        #     np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.beta))
        # )
        K1 = (self.R * (np.tan(self.beta))) / (
            np.cos(self.phi))
        K2 = np.tan(self.phi) * np.tan(self.beta)
        return (
            self.omega**2
            * K1
            * (1 + K2)
            * (K2 * (1 + np.sin(theta) * np.sin(theta)) + np.cos(theta))
            / ((1 + K2 * np.cos(theta)) ** 2)
        )

    def method1(self, theta):
        """
        First method where you can directly write displacement, velocity, and acceleration equations
        """
        # Space to write displacement equation
        displacement =  self._calculate_displacement(theta) # Your displacement equation

        # Space to write velocity equation
        velocity = self._calcualte_velocity(theta)  # Your velocity equation

        # Space to write acceleration equation
        acceleration = self._calcualte_acc(theta)  # Your acceleration equation

        return displacement, velocity, acceleration

    def method2(self, theta):
        """
        Second method using JAX to derive velocity and acceleration from displacement

        Example displacement equation:
        displacement = A * jnp.sin(omega * t)
        where:
        - A is amplitude
        - omega is angular frequency
        - t is time/theta
        """

        def displacement_equation(t):
            # Space to write your displacement equation
            # Example: displacement = 10 * jnp.sin(t)
            # N= (-self.L*(2*(1/jnp.cos(self.phi))+ (jnp.sin(t)*jnp.tan(self.beta)*jnp.tan(self.phi))) + (self.R* (jnp.sin(t)*jnp.tan(self.phi)+ jnp.cos(t)*jnp.tan(self.beta))))
            # D= (jnp.sin(t)*jnp.tan(self.beta)*jnp.tan(self.phi)+ 1)
            # N= (self.L- ((self.L+self.R*(jnp.cos(t)*jnp.tan(beta)))/(jnp.sin(self.phi)*jnp.cos(t)*jnp.tan(self.beta)+ jnp.cos(self.phi))))
            N = ((self.L - self.R * (jnp.cos(t) * jnp.tan(self.beta))) / (
                        jnp.tan(self.beta) * jnp.cos(t) * jnp.sin(self.phi) + jnp.cos(self.phi)))
            T1= (self.L / jnp.cos(self.phi))
            # displacement= (self.L * jnp.cos(t) * jnp.tan(self.beta) * jnp.tan(self.phi) + R * jnp.cos(t) * jnp.tan(self.beta)) / (
            #             (- jnp.cos(t) * jnp.tan(self.beta)) * jnp.sin(self.phi) + jnp.cos(self.phi))

            # displacement = ((jnp.cos(t) * jnp.tan(self.beta) * (self.R + self.L * jnp.tan(self.phi))) / (
            #             jnp.tan(self.beta) * jnp.cos(t) * jnp.sin(self.phi) - jnp.cos(self.phi)))- (self.L / jnp.cos(self.phi))
            # displacement = N - T1 # Your equation here
            disp = - (R * (jnp.tan(self.beta) /(jnp.cos(self.phi))) * ((1 - jnp.cos(theta)) /  (1 + (jnp.tan(self.beta) * jnp.tan(self.phi) * jnp.cos(theta)))))
            return disp

        # Create velocity and acceleration functions using JAX
        velocity_func = grad(displacement_equation)
        acceleration_func = grad(velocity_func)

        # Calculate all three values
        displacement = displacement_equation(theta)
        velocity = -self.omega*velocity_func(theta)
        acceleration = self.omega**2 * acceleration_func(theta)


        return float(displacement), float(velocity), float(acceleration)

    def compare_and_save(self, theta_range, filename='motion_comparison.csv'):
        """
        Compare both methods and save results to CSV
        """
        # Initialize lists for results
        disp1, vel1, acc1 = [], [], []
        disp2, vel2, acc2 = [], [], []

        # Calculate using both methods
        for t in theta_range:
            d1, v1, a1 = self.method1(t)
            d2, v2, a2 = self.method2(t)

            disp1.append(d1)
            vel1.append(v1)
            acc1.append(a1)
            disp2.append(d2)
            vel2.append(v2)
            acc2.append(a2)

        # Create DataFrame
        df = pd.DataFrame({
            'theta': theta_range,
            'displacement_method1': disp1,
            'velocity_method1': vel1,
            'acceleration_method1': acc1,
            'displacement_method2': disp2,
            'velocity_method2': vel2,
            'acceleration_method2': acc2
        })

        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

        return df

    def plot_comparison(self, df):
        """
        Plot comparison of both methods
        """
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

    def plot_save_acceleration(self, df, save_path='acceleration_comparison.png'):
        """
        Plot and save acceleration comparison between method 1 and method 2

        Parameters:
        df (pandas.DataFrame): DataFrame containing the acceleration data
        save_path (str): Path where the plot should be saved
        """
        plt.figure(figsize=(12, 8))

        plt.plot(df['theta'], df['acceleration_method1'], '-',
                 label='Using Hydraulic_motors_&_pump_theory', linewidth=2, color='blue')
        plt.plot(df['theta'], df['acceleration_method2'], ':',
                 label='Using_geometric_transformation_by_mit', linewidth=2, color='red')

        plt.xlabel('Theta (radians)', fontsize=12)
        plt.ylabel('Acceleration (m/s²)', fontsize=12)
        plt.title('Acceleration Comparison Between Methods', fontsize=14)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        plt.xticks(np.linspace(0, 2 * np.pi, 7),
                   ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])

        max_diff = np.max(np.abs(df['acceleration_method1'] - df['acceleration_method2']))
        mean_diff = np.mean(np.abs(df['acceleration_method1'] - df['acceleration_method2']))
        stats_text = f'Max Difference: {max_diff:.2f}\nMean Difference: {mean_diff:.2f}'
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                     fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_path}")
        plt.show()


def plot_save_acceleration(self, df, save_path='acceleration_comparison.png'):
    """
    Plot and save acceleration comparison between method 1 and method 2

    Parameters:
    df (pandas.DataFrame): DataFrame containing the acceleration data
    save_path (str): Path where the plot should be saved
    """
    # Create figure with appropriate size
    plt.figure(figsize=(12, 8))

    # Plot acceleration comparison with solid and dotted lines
    plt.plot(df['theta'], df['acceleration_method1'], '-',
             label='Using Hydraulic_motors_&_pump_theory', linewidth=2, color='blue')  # solid line
    plt.plot(df['theta'], df['acceleration_method2'], ':',
             label='Using_geometric_transformation_by_mit', linewidth=2, color='red')  # dotted line

    # Add labels and title
    plt.xlabel('Theta (radians)', fontsize=12)
    plt.ylabel('Acceleration (m/s²)', fontsize=12)
    plt.title('Acceleration Comparison Between Methods', fontsize=14)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Customize ticks
    plt.xticks(np.linspace(0, 2 * np.pi, 7),
               ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])

    # Add statistics annotation
    max_diff = np.max(np.abs(df['acceleration_method1'] - df['acceleration_method2']))
    mean_diff = np.mean(np.abs(df['acceleration_method1'] - df['acceleration_method2']))
    stats_text = f'Max Difference: {max_diff:.2f}\nMean Difference: {mean_diff:.2f}'
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                 fontsize=10, verticalalignment='top')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {save_path}")

    # Display the plot
    plt.show()


def plot_comprehensive_acceleration_analysis(results_phi5, results_phi0, save_path_base='acceleration_analysis'):
    """
    Create comprehensive plots showing both method differences and phi angle differences

    Parameters:
    results_phi5 (pandas.DataFrame): DataFrame containing acceleration data for phi=5°
    results_phi0 (pandas.DataFrame): DataFrame containing acceleration data for phi=0°
    save_path_base (str): Base path for saving plots
    """

    # Create two separate figures for better clarity
    fig_methods, (ax1) = plt.subplots(1, 1, figsize=(15, 12))
    # fig_angles, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 12))

    # === Figure 1: Method Differences (M1 vs M2) ===
    # Calculate differences between methods
    diff_methods_phi5 = results_phi5['acceleration_method1'] - results_phi0['acceleration_method2']
    diff_methods_phi0 = results_phi0['acceleration_method1'] - results_phi0['acceleration_method2']

    # # Plot for Method differences (phi=5°)
    # ax1.plot(results_phi5['theta'], results_phi5['acceleration_method1'], '-',
    #           linewidth=2, color='blue')
    # ax1.plot(results_phi5['theta'], results_phi0['acceleration_method1'], '-',
    #           linewidth=2, color='green')
    # ax1.set_xlabel('Theta (radians)', fontsize=12)
    # ax1.set_ylabel('Acceleration (m/s²)', fontsize=12)
    # ax1.set_title('Method  (φ=5 & 0°): M1 - M2', fontsize=14)

    # # Plot for Method differences (phi=0°)
    # ax1.plot(results_phi0['theta'], results_phi5['acceleration_method2'], '-',
    #           linewidth=2, color='red')
    # ax1.plot(results_phi5['theta'], results_phi0['acceleration_method2'], '-',
    #           linewidth=2, color='green')
    ax1.plot(results_phi5['theta'], results_phi5['acceleration_method1'], '-',
             linewidth=2, color='blue')
    ax1.plot(results_phi5['theta'], results_phi5['acceleration_method2'], '-',
             linewidth=2, color='yellow')
    ax1.set_xlabel('Theta (radians)', fontsize=12)
    ax1.set_ylabel('Acceleration Difference (m/s²)', fontsize=12)
    ax1.set_title('Method Differences (φ=0°): M1 - M2', fontsize=14)
    #
    # # === Figure 2: Phi Angle Differences (5° vs 0°) ===
    # # Calculate differences between phi angles
    # diff_phi_m1 = results_phi5['acceleration_method1'] - results_phi0['acceleration_method1']
    # diff_phi_m2 = results_phi5['acceleration_method2'] - results_phi0['acceleration_method2']
    #
    # # Plot for Phi differences (Method 1)
    # ax3.plot(results_phi5['theta'], diff_phi_m1, '-',
    #          label='φ(5°) - φ(0°) [M1]', linewidth=2, color='green')
    # ax3.set_xlabel('Theta (radians)', fontsize=12)
    # ax3.set_ylabel('Acceleration Difference (m/s²)', fontsize=12)
    # ax3.set_title('Phi Angle Differences (Method 1): φ(5°) - φ(0°)', fontsize=14)
    #
    # # Plot for Phi differences (Method 2)
    # ax4.plot(results_phi0['theta'], diff_phi_m2, '-',
    #          label='φ(5°) - φ(0°) [M2]', linewidth=2, color='purple')
    # ax4.set_xlabel('Theta (radians)', fontsize=12)
    # ax4.set_ylabel('Acceleration Difference (m/s²)', fontsize=12)
    # ax4.set_title('Phi Angle Differences (Method 2): φ(5°) - φ(0°)', fontsize=14)

    # # Common settings for all plots
    # for ax in [ax1, ax3, ax4]:
    #     # Add grid
    #     ax.grid(True, linestyle='--', alpha=0.7)
    #     # Add legend
    #     ax.legend(fontsize=10)
    #     # Add reference line
    #     ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    #     # Customize ticks
    #     one_rev = 2 * np.pi
    #     ax.set_xticks(np.linspace(0, one_rev, 7))
    #     ax.set_xticklabels(['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
    #
    #     # Calculate and add statistics
    #     if ax in [ax1]:
    #         diff_data = diff_methods_phi5 if ax == ax1 else diff_methods_phi0
    #     else:
    #         diff_data = diff_phi_m1 if ax == ax3 else diff_phi_m2
    #
    #     stats_text = (
    #         f'Max Difference: {np.max(np.abs(diff_data)):.2f} m/s²\n'
    #         f'Mean Difference: {np.mean(np.abs(diff_data)):.2f} m/s²\n'
    #         f'RMS Difference: {np.sqrt(np.mean(diff_data ** 2)):.2f} m/s²'
    #     )
    #     ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
    #             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
    #             fontsize=10, verticalalignment='top')
    #
    # # Adjust layouts
    # fig_methods.suptitle('Comparison of Methods (M1 vs M2)', fontsize=16, y=1.02)
    # fig_angles.suptitle('Comparison of Phi Angles (5° vs 0°)', fontsize=16, y=1.02)
    #
    # fig_methods.tight_layout()
    # fig_angles.tight_layout()

    # Display plots
    plt.show()


# Modified main execution block
if __name__ == "__main__":
    # Set parameters
    revolutions = 2
    rpm = 2000
    beta = 15
    phi_1 =3# First phi angle
    phi_2 = 0  # Second phi angle
    R = 44.6
    Bore_length = 70
    omega = 2 * np.pi * rpm / 60

    # Create theta range
    theta = np.linspace(0, 2 * np.pi * revolutions, 360)

    # Create analyzers and get results
    analyzer_1 = MotionAnalysis(R, beta, phi_1, omega, Bore_length)
    analyzer_2 = MotionAnalysis(R, beta, phi_2, omega, Bore_length)
    results_1 = analyzer_1.compare_and_save(theta)
    results_2 = analyzer_2.compare_and_save(theta)

    # Use the method correctly as part of the class
    analyzer_1.plot_save_acceleration(results_1, 'acceleration_comparison_phi1.png')
    analyzer_2.plot_save_acceleration(results_2, 'acceleration_comparison_phi2.png')

    analyzer_1

    # Generate comprehensive analysis plots
    plot_comprehensive_acceleration_analysis(results_1, results_2)
