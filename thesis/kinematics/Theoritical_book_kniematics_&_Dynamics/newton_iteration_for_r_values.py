import numpy as np
import matplotlib.pyplot as plt
import pint

ureg = pint.UnitRegistry()

class MotionAnalyzer:
    def __init__(self, r, alpha, phi, omega):
        # Convert inputs to magnitude if they're Pint quantities
        self.r = r.magnitude if hasattr(r, 'magnitude') else r
        self.alpha = alpha
        self.phi = phi
        self.omega = omega

    def _calculate_displacement(self, theta):
        """Helper method to calculate displacement"""
        K1 = (self.r * (np.tan(self.alpha))) / (
            np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.alpha))
        )
        K2 = np.tan(self.phi) * np.tan(self.alpha)
        return (K1*(1-np.cos(theta)))/(1+(K2* np.cos(theta)))

    def _calculate_velocity(self, theta):
        K1 = (self.r * (np.tan(self.alpha))) / (
            np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.alpha))
        )
        K2 = np.tan(self.phi) * np.tan(self.alpha)
        return K1 * self.omega * (1 + K2) * np.sin(theta) / (1 + K2 * np.cos(theta))

    def _calculate_acc(self, theta):
        K1 = (self.r * (np.tan(self.alpha))) / (
            np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.alpha))
        )
        K2 = np.tan(self.phi) * np.tan(self.alpha)
        return (
            self.omega**2
            * K1
            * (1 + K2)
            * (K2 * (1 + np.sin(theta) * np.sin(theta)) + np.cos(theta))
            / ((1 + K2 * np.cos(theta)) ** 2)
        )

def get_disp(R, theta, beta):
    if hasattr(R, 'magnitude'):
        R = R.magnitude
    y = R * np.cos(theta)
    b = R - y
    z = b * np.tan(beta)
    return -z

def newtons_method_incline(Rmin, Rmax, phi, y_min, y_max, L, tolerance=1e-6, max_iter=250):
    y_positions = np.linspace(y_min.magnitude, y_max.magnitude, num=250) * y_min.units
    R_values = []

    for y in y_positions:
        # Better initial guess: use a value between Rmin and Rmax
        R = (Rmin + Rmax) / 2  # Start from middle point

        for i in range(max_iter):
            # Calculate function value (the equation we're trying to solve)
            f_value = R - (Rmin + np.tan(phi) * (y - y_min))
            # Derivative of the function with respect to R
            f_derivative = 1.0  # Since d/dR(R) = 1

            # Newton's method iteration
            R_new = R - f_value / f_derivative

            # Check if R_new is within bounds
            if R_new < Rmin:
                R_new = Rmin
            elif R_new > Rmax:
                R_new = Rmax

            # Check convergence
            if abs(R_new - R) < tolerance * Rmin.units:
                R_values.append(R_new.magnitude)  # Store just the magnitude
                break

            R = R_new
        else:
            # If no convergence, use the last calculated value
            R_values.append(R.magnitude)  # Store just the magnitude
            print(f"Warning: Newton's method did not fully converge for y = {y}, using last iteration value")

    return y_positions, np.array(R_values) * Rmin.units


def runge_kutta_incline(Rmin, Rmax, phi, y_min, y_max, L, step_size=0.01):
    """
    Solve for R values using 4th order Runge-Kutta method
    """
    y_positions = np.linspace(y_min.magnitude, y_max.magnitude, num=250) * y_min.units
    R_values = []

    def dR_dy(y, R):
        """Differential equation: dR/dy = tan(phi)"""
        return np.tan(phi)

    def rk4_step(y, R, h):
        """Single step of RK4 method"""
        k1 = dR_dy(y, R)
        k2 = dR_dy(y + h / 2, R + h * k1 / 2)
        k3 = dR_dy(y + h / 2, R + h * k2 / 2)
        k4 = dR_dy(y + h, R + h * k3)

        return R + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Initial condition
    R = Rmin

    for y in y_positions:
        # RK4 integration
        h = step_size * y_min.units  # Step size with units

        # Ensure R stays within bounds
        R = min(max(R, Rmin), Rmax)
        R_values.append(R.magnitude)

        # Update R for next step
        R = rk4_step(y, R, h)

    return y_positions, np.array(R_values) * Rmin.units


def create_analysis_plots(y_positions, R_values, theta, displacement_values, velocity_values,
                          acceleration_values, R_magnitudes, beta, phi):
    # Create a figure with a custom layout
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])

    # 1. R values along incline with visualization
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(y_positions.magnitude, R_values.magnitude,
                     c=R_values.magnitude, cmap='viridis',
                     label='R values')
    ax1.set_title('R Distribution Along Inclined Block', fontsize=12, pad=20)
    ax1.set_xlabel('y-coordinate [m]')
    ax1.set_ylabel('R [m]')
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(sc, ax=ax1, label='R magnitude [m]')

    # # 2. 3D visualization of motion path
    # ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    # theta_mesh, y_mesh = np.meshgrid(theta, y_positions.magnitude)
    # z_mesh = z_values
    # ax2.plot_surface(theta_mesh, y_mesh, z_mesh, cmap='viridis', alpha=0.8)
    # ax2.set_title('3D Motion Path', fontsize=12, pad=20)
    # ax2.set_xlabel('Theta [rad]')
    # ax2.set_ylabel('y [m]')
    # ax2.set_zlabel('z [m]')

    # 3. Displacement analysis
    ax3 = fig.add_subplot(gs[1, 0])
    theta_deg = np.degrees(theta)
    for i, R in enumerate(R_magnitudes):
        ax3.plot(theta_deg, displacement_values[i],
                 label=f'R={R:.2f}m', alpha=0.7)
    ax3.set_title('Displacement vs Theta\nfor Different R Values', fontsize=12, pad=20)
    ax3.set_xlabel('Theta [degrees]')
    ax3.set_ylabel('Displacement [m]')
    ax3.grid(True, linestyle='--', alpha=0.7)
    if len(R_magnitudes) <= 5:
        ax3.legend()

    # 4. Velocity profile
    ax4 = fig.add_subplot(gs[1, 1])
    for i, R in enumerate(R_magnitudes):
        ax4.plot(theta_deg, velocity_values[i],
                 label=f'R={R:.2f}m', alpha=0.7)
    ax4.set_title('Velocity Profile vs Theta', fontsize=12, pad=20)
    ax4.set_xlabel('Theta [degrees]')
    ax4.set_ylabel('Velocity [m/s]')
    ax4.grid(True, linestyle='--', alpha=0.7)

    # 5. Acceleration analysis
    ax5 = fig.add_subplot(gs[2, 0])
    for i, R in enumerate(R_magnitudes):
        ax5.plot(theta_deg, acceleration_values[i],
                 label=f'R={R:.2f}m', alpha=0.7)
    ax5.set_title('Acceleration vs Theta', fontsize=12, pad=20)
    ax5.set_xlabel('Theta [degrees]')
    ax5.set_ylabel('Acceleration [m/s²]')
    ax5.grid(True, linestyle='--', alpha=0.7)

    # 6. Phase space plot
    ax6 = fig.add_subplot(gs[2, 1])
    for i, R in enumerate(R_magnitudes):
        ax6.plot(displacement_values[i], velocity_values[i],
                 label=f'R={R:.2f}m', alpha=0.7)
    ax6.set_title('Phase Space Plot\n(Velocity vs Displacement)', fontsize=12, pad=20)
    ax6.set_xlabel('Displacement [m]')
    ax6.set_ylabel('Velocity [m/s]')
    ax6.grid(True, linestyle='--', alpha=0.7)

    # 7. Angular analysis
    ax7 = fig.add_subplot(gs[3, 0])
    angular_velocity = omega * np.ones_like(theta)  # constant angular velocity
    angular_acceleration = np.zeros_like(theta)  # zero angular acceleration
    ax7.plot(theta_deg, angular_velocity, label='Angular Velocity', color='blue')
    ax7.plot(theta_deg, angular_acceleration, label='Angular Acceleration', color='red')
    ax7.set_title('Angular Motion Analysis', fontsize=12, pad=20)
    ax7.set_xlabel('Theta [degrees]')
    ax7.set_ylabel('Angular Velocity [rad/s] / Acceleration [rad/s²]')
    ax7.grid(True, linestyle='--', alpha=0.7)
    ax7.legend()

    # 8. Geometric parameters
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')
    info_text = (
        f"Geometric Parameters:\n\n"
        f"Inclination angle (φ): {np.degrees(phi):.1f}°\n"
        f"Beta angle (β): {np.degrees(beta):.1f}°\n"
        f"R range: [{Rmin.magnitude:.2f}, {Rmax.magnitude:.2f}] m\n"
        f"y range: [{y_min.magnitude:.2f}, {y_max.magnitude:.2f}] m\n"
        f"Angular velocity (ω): {omega:.1f} rad/s\n"
        f"Number of R values: {len(R_magnitudes)}\n"
        f"Number of theta points: {len(theta)}\n"
    )
    ax8.text(0.1, 0.9, info_text, transform=ax8.transAxes,
             fontfamily='monospace', fontsize=10)

    # Adjust layout
    plt.tight_layout()
    return fig


# Parameters with units
Rmin = 0.0 * ureg.meter
Rmax = 2.0 * ureg.meter
phi = np.pi / 15
y_min = 0.0 * ureg.meter
y_max = 2.0 * ureg.meter
L = (y_max - y_min)
beta = np.radians(15)
alpha = np.radians(15)
omega = 10.0  # rad/s

# Create theta array
theta = np.linspace(0, 2 * np.pi, 360)

# Get R values
# y_positions, R_values = newtons_method_incline(Rmin, Rmax, phi, y_min, y_max, L)
y_positions, R_values = runge_kutta_incline(Rmin, Rmax, phi, y_min, y_max, L)
# Calculate motion parameters for each R value
# Convert R_values to magnitude for calculations
R_magnitudes = R_values.magnitude

displacement_values = np.zeros((len(R_magnitudes), len(theta)))
velocity_values = np.zeros((len(R_magnitudes), len(theta)))
acceleration_values = np.zeros((len(R_magnitudes), len(theta)))

for i, R in enumerate(R_magnitudes):
    analyzer = MotionAnalyzer(R, alpha, phi, omega)
    for j, t in enumerate(theta):
        displacement_values[i, j] = analyzer._calculate_displacement(t)
        velocity_values[i, j] = analyzer._calculate_velocity(t)
        acceleration_values[i, j] = analyzer._calculate_acc(t)


# After calculating all values, create the plots:
fig = create_analysis_plots(
    y_positions=y_positions,
    R_values=R_values,
    theta=theta,
    displacement_values=displacement_values,
    velocity_values=velocity_values,
    acceleration_values=acceleration_values,

    R_magnitudes=R_magnitudes,
    beta=beta,
    phi=phi
)

# Save the figure
plt.savefig('motion_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
