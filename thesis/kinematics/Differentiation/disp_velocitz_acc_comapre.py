import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import grad



def collect_accleation_angles_from_transformation(
        bore_length, beta, phi, theta, speed, R
):
    acc = []


    omega = 2 * np.pi * speed / 60

    # Convert angles from degrees to radians
    beta_rad = np.deg2rad(beta)
    phi_rad = np.deg2rad(phi)

    for t in theta:
        acc1 = (
                       bore_length * (np.cos(t) * np.tan(beta_rad) * np.tan(phi_rad))
                       - R * ((np.cos(t) * np.tan(phi_rad)) - (np.sin(t) * np.tan(beta_rad)))
               ) * (np.cos(t) * np.tan(beta_rad) * np.tan(phi_rad)) - (
                       (np.sin(t) * np.tan(beta_rad) * np.tan(phi_rad) + 1)
                       * (
                               bore_length * (np.sin(t) * np.tan(beta_rad) * np.tan(phi_rad))
                               - R * ((np.sin(t) * np.tan(phi_rad)) + (np.cos(t) * np.tan(beta_rad)))
                       )
               )
        T1_acc = acc1 / ((np.sin(t) * np.tan(beta_rad) * np.tan(phi_rad) + 1) ** 2)

        acc2 = (np.sin(t) * np.tan(beta_rad) * np.tan(phi_rad)) ** 2 * (
                (
                        bore_length * (np.cos(t) * np.tan(beta_rad) * np.tan(phi_rad))
                        - R * ((np.cos(t) * np.tan(phi_rad)) - (np.sin(t) * np.tan(beta_rad)))
                )
                * (np.cos(t) * np.tan(beta_rad) * np.tan(phi_rad))
                + (
                        (np.sin(t) * np.tan(beta_rad) * np.tan(phi_rad))
                        * (
                                bore_length
                                * (2 * (1 / np.cos(phi_rad)) + (np.sin(t) * np.tan(beta_rad) * np.tan(phi_rad)))
                                + R * (np.sin(t) * np.tan(phi_rad) + np.cos(t) * np.tan(beta_rad))
                        )
                )
        ) + 2 * (np.sin(t) * np.tan(beta_rad) * np.tan(phi_rad) + 1) * (
                       np.cos(t) * np.tan(beta_rad) * np.tan(phi_rad)
               ) ** 2 * (
                       bore_length
                       * (2 * (1 / np.cos(phi_rad)) + (np.sin(t) * np.tan(beta_rad) * np.tan(phi_rad)))
                       - R * (np.sin(t) * np.tan(phi_rad) + np.cos(t) * np.tan(beta_rad))
               )
        T2_acc = acc2 / ((np.sin(t) * np.tan(beta_rad) * np.tan(phi_rad) + 1) ** 4)



        acc.append(((omega ** 2) * (T2_acc+ T1_acc)))

    acc_array = np.array(acc)
    print(f"Maximum acceleration: {np.max(acc_array):.2f}")
    print(f"Minimum acceleration: {np.min(acc_array):.2f}")

    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    plt.plot(theta, acc)
    plt.xlabel("Theta (radians)")
    plt.ylabel("Acceleration")
    plt.title("Acceleration vs Theta")
    plt.grid(True)
    plt.show()



# Create acceleration function by taking derivative of velocity with respect to t


if __name__ == "__main__":
    revolutions = 5
    phi = 5
    beta = 15
    borelength = 10
    speed = 2000
    n_points = 360 * revolutions
    theta = np.linspace(0, 2 * np.pi * revolutions, n_points)
    radius = 43.5

    collect_accleation_angles_from_transformation(borelength, beta, phi, theta, speed, radius)