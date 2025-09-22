import numpy as np
import matplotlib.pyplot as plt

# Define constants (you can change these as needed)
alpha_deg = 15# Swash plate tilt angle α in degrees
beta_deg = 0   # Swash plate cross angle β in degrees
gamma_deg = 0   # Piston inclination γ in degrees
R =131.46       # Distribution circle radius (ρ)
L1 = 54          # Distance from coordinate origin to valve plate (h1)

# Convert angles to radians
alpha = np.radians(alpha_deg)
beta = np.radians(beta_deg)
gamma = np.radians(gamma_deg)

# Define crank angle φ from 0 to 2π
phi = np.linspace(0, 2 * np.pi, 1000)

# Displacement function
numerator = (R + L1 * np.tan(gamma)) * (np.tan(alpha) * np.cos(phi) - np.tan(beta) * np.sin(phi))
denominator = np.cos(gamma) - np.tan(alpha) * np.sin(gamma) * np.cos(phi) + np.tan(beta) * np.sin(gamma) * np.sin(phi)
x = numerator / denominator

num = R * np.tan(alpha) * np.sin(phi)
num1= R* np.tan(alpha) * np. sin(phi +270)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(phi), num)
plt.plot(np.degrees(phi), num1)
plt.plot(np.degrees(phi), x)
plt.title('Piston Displacement vs. Crank Angle')
plt.xlabel('Crank Angle φ (degrees)')
plt.ylabel('Displacement x')
plt.grid(True)
plt.tight_layout()
plt.show()
