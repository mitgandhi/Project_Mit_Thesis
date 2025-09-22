import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from dataclasses import dataclass


@dataclass
class OperatingPistonGap:
    beta_rad: float
    betamax_rad: float
    gamma_rad: float
    phi_rad: float
    g_rad: float
    L1: float


@dataclass
class GeometryPistonGap:
    sK: float = 0.0
    sK_alt: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    velocity_alt: float = 0.0
    acceleration_alt: float = 0.0


class CPistonGap:
    def __init__(self):
        self.operatingpistongap = OperatingPistonGap(
            beta_rad=0.0,
            betamax_rad=0.0,
            gamma_rad=0.0,
            phi_rad=0.0,
            g_rad=0.0,
            L1=0.0
        )
        self.geometrypistongap = GeometryPistonGap()
        self.rB = 0.0

        # Create symbolic variables for derivatives
        self.phi = sp.Symbol('phi')
        self.t = sp.Symbol('t')
        self.omega = 2000 * 2 * np.pi / 60  # Convert RPM to rad/s

        # Define phi as a function of time
        self.phi_t = self.omega * self.t

    def calculate_derivatives_method1(self):
        beta = self.operatingpistongap.beta_rad
        betamax = self.operatingpistongap.betamax_rad
        gamma = self.operatingpistongap.gamma_rad
        R = self.rB

        # Define sK expression symbolically
        phi_odp = 0 if beta == 0 else -sp.atan(sp.tan(gamma) / sp.sin(beta))
        delta_psi = (R * sp.tan(beta) * (1 - sp.cos(phi_odp)) +
                     R * sp.tan(gamma) * sp.sin(phi_odp) / sp.cos(beta))

        sK_expr = (-R * sp.tan(beta) * (1 - sp.cos(self.phi)) -
                   R * sp.tan(gamma) * sp.sin(self.phi) / sp.cos(beta) -
                   R * (sp.tan(betamax) - sp.tan(beta)) +
                   delta_psi)

        # Calculate velocity and acceleration expressions
        sK_t = sK_expr.subs(self.phi, self.phi_t)
        velocity = sp.diff(sK_t, self.t)
        acceleration = sp.diff(velocity, self.t)

        return sK_expr, velocity, acceleration

    def calculate_derivatives_method2(self):
        gamma = self.operatingpistongap.gamma_rad
        beta = self.operatingpistongap.beta_rad
        betamax = self.operatingpistongap.betamax_rad
        g = self.operatingpistongap.g_rad
        L1 = self.operatingpistongap.L1
        R = self.rB

        # Define alternative sK expression symbolically
        numerator = (L1 - R * sp.sin(self.phi) * sp.tan(gamma) -
                     R * sp.cos(self.phi) * sp.tan(beta))
        denominator = (+sp.sin(g) * sp.sin(self.phi) * sp.tan(gamma) +
                       sp.sin(g) * sp.cos(self.phi) * sp.tan(beta) +
                       sp.cos(g))

        sK_basic = (numerator / denominator) - (L1 / sp.cos(g))
        offset = -R * (sp.tan(betamax) - sp.tan(beta)) / sp.cos(g)

        phi_odp = 0 if beta == 0 and g == 0 else -sp.atan(sp.tan(gamma) / (sp.sin(g) * sp.sin(beta)))
        initial_offset = (R * sp.tan(g) * sp.tan(beta) * (1 - sp.cos(phi_odp)) +
                          R * sp.tan(gamma) * sp.sin(phi_odp) / (sp.cos(g) * sp.cos(beta)))

        sK_expr = -(sK_basic + offset + initial_offset)

        # Calculate velocity and acceleration expressions
        sK_t = sK_expr.subs(self.phi, self.phi_t)
        velocity = -sp.diff(sK_t, self.t)
        acceleration = sp.diff(velocity, self.t)

        return sK_expr, velocity, acceleration


def plot_kinematics():
    piston_gap = CPistonGap()

    # Set parameters
    piston_gap.rB = 0.05
    piston_gap.operatingpistongap.beta_rad = math.radians(0)
    piston_gap.operatingpistongap.betamax_rad = math.radians(15)
    piston_gap.operatingpistongap.gamma_rad = math.radians(5)
    piston_gap.operatingpistongap.g_rad = math.radians(5)
    piston_gap.operatingpistongap.L1 = 0.1

    # Get symbolic expressions
    sK1_expr, v1_expr, a1_expr = piston_gap.calculate_derivatives_method1()
    sK2_expr, v2_expr, a2_expr = piston_gap.calculate_derivatives_method2()

    # Create arrays for plotting
    t_values = np.linspace(0, 2 * np.pi / piston_gap.omega, 360)
    phi_values = piston_gap.omega * t_values

    # Calculate numerical values
    v1_values = [float(v1_expr.subs(piston_gap.t, t)) for t in t_values]
    a1_values = [float(a1_expr.subs(piston_gap.t, t)) for t in t_values]
    v2_values = [float(v2_expr.subs(piston_gap.t, t)) for t in t_values]
    a2_values = [float(a2_expr.subs(piston_gap.t, t)) for t in t_values]

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Velocity plot
    ax1.plot(np.degrees(phi_values), v1_values, 'b-', label='Method 1')
    ax1.plot(np.degrees(phi_values), v2_values, 'r--', label='Method 2')
    ax1.set_title('Velocity Comparison')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.grid(True)
    ax1.legend()

    # Acceleration plot
    ax2.plot(np.degrees(phi_values), a1_values, 'b-', label='Method 1')
    ax2.plot(np.degrees(phi_values), a2_values, 'r--', label='Method 2')
    ax2.set_title('Acceleration Comparison')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return phi_values, v1_values, a1_values, v2_values, a2_values


if __name__ == "__main__":
    phi_values, v1_values, a1_values, v2_values, a2_values = plot_kinematics()

    print("\nMethod 1 Kinematics:")
    print(f"Maximum velocity: {max(v1_values):.6f} m/s")
    print(f"Minimum velocity: {min(v1_values):.6f} m/s")
    print(f"Maximum acceleration: {max(a1_values):.6f} m/s²")
    print(f"Minimum acceleration: {min(a1_values):.6f} m/s²")

    print("\nMethod 2 Kinematics:")
    print(f"Maximum velocity: {max(v2_values):.6f} m/s")
    print(f"Minimum velocity: {min(v2_values):.6f} m/s")
    print(f"Maximum acceleration: {max(a2_values):.6f} m/s²")
    print(f"Minimum acceleration: {min(a2_values):.6f} m/s²")