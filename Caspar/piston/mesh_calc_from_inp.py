import numpy as np
import matplotlib.pyplot as plt
from calculate_dF import PistonGap, GeometryPistonGap, OperatingPistonGap
from utils import MeshReader, MeshProcessor, MeshVisualizer


def process_mesh(N, M, Q, rK, L):
    """Create a cylindrical mesh"""
    r = np.linspace(0.98 * rK, 1.02 * rK, N)
    theta = np.linspace(0, 2 * np.pi, M)
    z = np.linspace(0, L, Q)
    theta_mg, r_mg, z_mg = np.meshgrid(theta, r, z)

    X = r_mg * np.cos(theta_mg)
    Y = r_mg * np.sin(theta_mg)
    Z = z_mg
    dAz = r_mg * (theta[1] - theta[0]) * (r[1] - r[0])

    return X, Y, Z, r_mg, theta_mg, z_mg, dAz


def generate_fields(N, M, Q, Theta, operating):
    """Generate pressure and gap height fields"""
    p = np.zeros((N, M, Q))
    h1 = np.zeros((N, M, Q))

    for k in range(Q):
        base_pressure = operating.pDC - (operating.pDC - operating.pCase) * k / (Q - 1)
        taper_factor = 1 + 0.1 * k / (Q - 1)

        for i in range(N):
            for j in range(M):
                theta = Theta[i, j, k]
                p[i, j, k] = base_pressure * (1 + 0.1 * np.cos(2 * theta))
                h1[i, j, k] = 10e-6 * (1 + 0.2 * np.cos(theta)) * taper_factor

    return p, h1


def setup_piston_gap(geometry_params, operating_params):
    """Setup PistonGap instance with given parameters"""
    geometry = GeometryPistonGap(**geometry_params)
    operating = OperatingPistonGap(**operating_params)
    return PistonGap(geometry, operating)


def plot_results(X, Y, p, h1, sigma, mid_Q):
    """Plot pressure, gap height, and contact stress distributions"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plots = [
        (p[:, :, mid_Q] / 1e6, 'Pressure [MPa]'),
        (h1[:, :, mid_Q] * 1e6, 'Gap Height [µm]'),
        (sigma / 1e6, 'Contact Stress [MPa]')
    ]

    for ax, (data, title) in zip([ax1, ax2, ax3], plots):
        im = ax.contourf(X[:, :, mid_Q], Y[:, :, mid_Q], data, levels=20)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def print_forces(forces, dF):
    """Print force calculation results"""
    print("\n=== Force Calculation Results ===")
    print(f"\nFluid Forces:\nFfK = {forces.FfK:.2e} N\nFfKx = {forces.FfKx:.2e} N")
    print(f"FfKy = {forces.FfKy:.2e} N\nMfKx = {forces.MfKx:.2e} N⋅m\nMfKy = {forces.MfKy:.2e} N⋅m")

    print(f"\nContact Forces:\nFcK = {forces.FcK:.2e} N\nFcKx = {forces.FcKx:.2e} N")
    print(f"FcKy = {forces.FcKy:.2e} N\nMcKx = {forces.McKx:.2e} N⋅m\nMcKy = {forces.McKy:.2e} N⋅m")

    print("\nForce Differences:")
    for i, force in enumerate(dF):
        print(f"dF[{i}] = {force:.2e} N")


def main():
    # Mesh parameters
    N, M, Q = 30, 60, 20
    rK = 0.02  # 20mm radius
    L = 0.05  # 50mm length

    geometry_params = {
        'mK': 0.5, 'lSK': 0.05, 'zRK': 0.02,
        'lvar': 0.03, 'hmin': 5e-6
    }

    operating_params = {
        'beta_rad': np.radians(15),
        'betamax_rad': np.radians(15),
        'gamma_rad': np.radians(0),
        'phi_rad': np.radians(360),
        'omega': 1000* np.pi/180,
        'pDC': 200e5,
        'pCase': 1e5,
        'vK': 1.0
    }

    # Create mesh and setup piston
    X, Y, Z, R, Theta, Z_mg, dAz = process_mesh(N, M, Q, rK, L)
    piston = setup_piston_gap(geometry_params, operating_params)
    piston.initialize_mesh(N, M, Q)

    # Set geometric parameters
    piston.AreaK = np.pi * rK ** 2
    piston.rB = 0.05
    piston.rK = rK
    piston.rZ = 0.04

    # Generate fields and set mid-plane values
    p, h1 = generate_fields(N, M, Q, Theta, piston.operating)
    mid_Q = Q // 2
    piston.p = p[:, :, mid_Q]
    piston.h1 = h1[:, :, mid_Q]
    piston.dAz = dAz[:, :, mid_Q]
    piston.phi = Theta[:, :, mid_Q]
    piston.zKj = Z_mg[:, :, mid_Q]

    # Calculate forces
    piston.calc_fluid_forces()
    piston.calc_contact_forces()
    dF = piston.calc_force_difference()

    # Output results
    print_forces(piston.forces, dF)
    plot_results(X, Y, p, h1, piston.sigma, mid_Q)


if __name__ == "__main__":
    main()