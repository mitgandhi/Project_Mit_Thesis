import math
import matplotlib.pyplot as plt
import numpy as np


def piston_calcs_k(r_B, beta_rad, betamax_rad, gamma_rad, phi_deg):
    """
    Calculate piston stroke based on the given parameters.

    Parameters:
    - r_B: radius B parameter
    - beta_rad: beta angle in radians
    - betamax_rad: maximum beta angle in radians
    - gamma_rad: gamma angle in radians
    - phi_deg: phi angle in degrees (0-360)

    Returns:
    - s_K: calculated piston stroke
    """
    # Convert phi from degrees to radians
    phi_rad = math.radians(phi_deg)

    # Calculate phi_odp (offset displacement position)
    if beta_rad == 0:
        phi_odp = 0
    else:
        phi_odp = -math.atan(math.tan(gamma_rad) / math.sin(beta_rad))

    # Calculate delta_psi (offset in s_K due to gamma angle)
    delta_psi = r_B * math.tan(beta_rad) * (1 - math.cos(phi_odp)) + \
                r_B * math.tan(gamma_rad) * math.sin(phi_odp) / math.cos(beta_rad)

    # Calculate components of s_K
    s_K_beta = -r_B * math.tan(beta_rad) * (1 - math.cos(phi_rad))
    s_K_gamma = -r_B * math.tan(gamma_rad) * math.sin(phi_rad) / math.cos(beta_rad)

    # Calculate final s_K value
    s_K = s_K_beta + s_K_gamma - r_B * (math.tan(betamax_rad) - math.tan(beta_rad)) + delta_psi

    return s_K


# Example usage:
if __name__ == "__main__":
    # These values need to be set manually
    r_B = 0.042  # Example value, replace with your value
    beta_rad = math.radians(12)  # Example value in degrees, converted to radians
    betamax_rad = math.radians(12)  # Example value in degrees, converted to radians
    gamma_rad = math.radians(0)  # Example value in degrees, converted to radians

    # Calculate for full 0-360 range in steps of 1 degree
    phi_values = np.arange(0, 361, 1)  # 0 to 360 in steps of 1 degree
    s_K_values = [piston_calcs_k(r_B, beta_rad, betamax_rad, gamma_rad, phi) for phi in phi_values]

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(phi_values, s_K_values, 'b-', linewidth=2)
    plt.title('Piston Stroke (s_K) vs. Phi Angle')
    plt.xlabel('Phi Angle (degrees)')
    plt.ylabel('Piston Stroke (s_K)')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Add markers at key points (0, 90, 180, 270, 360 degrees)
    key_points = [0, 90, 180, 270, 360]
    key_values = [piston_calcs_k(r_B, beta_rad, betamax_rad, gamma_rad, phi) for phi in key_points]
    plt.plot(key_points, key_values, 'ro', markersize=6)

    for phi, s_K in zip(key_points, key_values):
        plt.annotate(f'({phi}°, {s_K:.4f})',
                     xy=(phi, s_K),
                     xytext=(5, 5),
                     textcoords='offset points')

    # Find and mark min/max values
    min_s_K = min(s_K_values)
    max_s_K = max(s_K_values)
    min_phi = phi_values[s_K_values.index(min_s_K)]
    max_phi = phi_values[s_K_values.index(max_s_K)]

    plt.plot([min_phi], [min_s_K], 'gs', markersize=8)
    plt.plot([max_phi], [max_s_K], 'gs', markersize=8)
    plt.annotate(f'Min: ({min_phi}°, {min_s_K:.4f})',
                 xy=(min_phi, min_s_K),
                 xytext=(5, -15),
                 textcoords='offset points')
    plt.annotate(f'Max: ({max_phi}°, {max_s_K:.4f})',
                 xy=(max_phi, max_s_K),
                 xytext=(5, 5),
                 textcoords='offset points')

    # Save the plot if needed
    # plt.savefig('piston_stroke_plot.png', dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    # Print some key values
    print(f"Min s_K: {min_s_K:.6f} at phi = {min_phi}°")
    print(f"Max s_K: {max_s_K:.6f} at phi = {max_phi}°")
    print(f"Range: {max_s_K - min_s_K:.6f}")