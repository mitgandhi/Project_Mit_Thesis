import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
import pandas as pd


def velocity(t, phi, beta, L, R):
    """
    Compute velocity given parameters
    """
    denominator = (jnp.tan(phi) * jnp.sin(t) * jnp.tan(beta) + 1) ** 2

    term1 = -((L * (jnp.cos(t) * jnp.tan(beta) * jnp.tan(phi)) -
               R * (jnp.cos(t) * jnp.tan(phi) - jnp.sin(t) * jnp.tan(beta))) *
              (jnp.tan(phi) * jnp.sin(t) * jnp.tan(beta) + 1)) / denominator

    term2 = (L * (2 * (1 / jnp.cos(phi)) + jnp.sin(t) * jnp.tan(beta) * jnp.tan(phi)) -
             R * (jnp.sin(t) * jnp.tan(phi) + jnp.cos(t) * jnp.tan(beta))) * \
            (jnp.cos(t) * jnp.tan(beta) * jnp.tan(phi)) / denominator

    return term1 + term2


# Create acceleration function
acceleration_func = grad(velocity, argnums=0)


def collect_motion_data(L, beta, phi, theta, speed, R):
    vel = []
    acc = []
    omega = 2 * np.pi * speed / 60

    # Convert angles from degrees to radians
    beta_rad = np.deg2rad(beta)
    phi_rad = np.deg2rad(phi)

    # Convert theta to degrees for CSV
    theta_degrees = np.rad2deg(theta) % 360

    for t in theta:
        # Calculate velocity
        vel_value = velocity(float(t), float(phi_rad), float(beta_rad), float(L), float(R))
        vel.append(float(omega * vel_value))

        # Calculate acceleration
        acc_value = acceleration_func(float(t), float(phi_rad), float(beta_rad), float(L), float(R))
        acc.append(float(omega ** 2 * acc_value))

    vel_array = np.array(vel)
    acc_array = np.array(acc)

    # Create DataFrame
    df = pd.DataFrame({
        'degree': theta_degrees,
        'velocity': vel_array,
        'acceleration': acc_array
    })

    # Save to CSV
    df.to_csv('motion_data.csv', index=False)

    print(f"Data saved to motion_data.csv")
    print(f"Velocity - Max: {np.max(vel_array):.2f}, Min: {np.min(vel_array):.2f}")
    print(f"Acceleration - Max: {np.max(acc_array):.2f}, Min: {np.min(acc_array):.2f}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot velocity
    ax1.plot(theta, vel, 'b-', label='Velocity')
    ax1.set_xlabel('Theta (radians)')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Velocity vs Theta')
    ax1.grid(True)
    ax1.legend()

    # Plot acceleration
    ax2.plot(theta, acc, 'r-', label='Acceleration')
    ax2.set_xlabel('Theta (radians)')
    ax2.set_ylabel('Acceleration')
    ax2.set_title('Acceleration vs Theta')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    revolutions = 5
    phi = 5
    beta = 15
    borelength = 10
    speed = 2000
    n_points = 360 * revolutions  # One point per degree
    theta = np.linspace(0, 2 * np.pi * revolutions, n_points)
    radius = 43.5

    collect_motion_data(
        borelength, beta, phi, theta, speed, radius
    )