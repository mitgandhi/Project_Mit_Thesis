import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify

# Define symbolic variables
phi, beta, gamma, r_max, L, R = sp.symbols('phi beta gamma r_max L R')

# Define the radius function for inclined piston
r_inclined = r_max - (2 * L * (1 - sp.cos(gamma)) * (sp.sin(phi/2))**2)

# Define straight piston radius
r_straight = r_max

# Define K constant from the new formula
K = sp.tan(beta) * sp.tan(gamma)

# Define the new displacement formula from the image
displacement_new = -R * (sp.tan(beta)/sp.cos(gamma)) * (1-sp.cos(phi))/(1+(K*sp.cos(phi)))

# Define old displacement formula with inclined radius
s_K_inclined = -r_inclined * sp.tan(beta) * (1 - sp.cos(phi))

# Define straight piston displacement formula
s_K_straight = -r_straight * sp.tan(beta) * (1 - sp.cos(phi))

# Calculate velocity and acceleration for all formulas (first and second derivatives with respect to phi)
# For the inclined piston formula
v_K_inclined = sp.diff(s_K_inclined, phi)
a_K_inclined = sp.diff(v_K_inclined, phi)

# For the straight piston formula
v_K_straight = sp.diff(s_K_straight, phi)
a_K_straight = sp.diff(v_K_straight, phi)

# For the new formula
v_new = sp.diff(displacement_new, phi)
a_new = sp.diff(v_new, phi)

# Convert symbolic expressions to numerical functions
def create_lambda_function(expr):
    return lambdify((phi, beta, gamma, r_max, L, R), expr, modules=['numpy'])

# Create numerical functions for the inclined piston formula
s_K_inclined_func = create_lambda_function(s_K_inclined)
v_K_inclined_func = create_lambda_function(v_K_inclined)
a_K_inclined_func = create_lambda_function(a_K_inclined)

# Create numerical functions for the new formula
displacement_new_func = create_lambda_function(displacement_new)
v_new_func = create_lambda_function(v_new)
a_new_func = create_lambda_function(a_new)

# Create numerical functions for the straight piston formula
s_K_straight_func = create_lambda_function(s_K_straight)
v_K_straight_func = create_lambda_function(v_K_straight)
a_K_straight_func = create_lambda_function(a_K_straight)

# Set parameter values
beta_val = np.radians(14)  # swashplate angle in radians
gamma_val = np.radians(5)  # piston inclination angle in radians
r_max_val = 42.08  # mm
L_val = 20.13 # mm
R_val = r_max_val  # Using r_max as R for the new formula

# Generate phi values (crank angle)
phi_vals = np.linspace(0, 2*np.pi, 1000)

# Calculate values for plotting using the inclined piston formula
s_inclined = s_K_inclined_func(phi_vals, beta_val, gamma_val, r_max_val, L_val, R_val)
v_inclined = v_K_inclined_func(phi_vals, beta_val, gamma_val, r_max_val, L_val, R_val)
a_inclined = a_K_inclined_func(phi_vals, beta_val, gamma_val, r_max_val, L_val, R_val)

# Calculate values for plotting using the new formula
s_new = displacement_new_func(phi_vals, beta_val, gamma_val, r_max_val, L_val, R_val)
v_new = v_new_func(phi_vals, beta_val, gamma_val, r_max_val, L_val, R_val)
a_new = a_new_func(phi_vals, beta_val, gamma_val, r_max_val, L_val, R_val)

# Calculate values for plotting using the straight piston formula
s_straight = s_K_straight_func(phi_vals, beta_val, gamma_val, r_max_val, L_val, R_val)
v_straight = v_K_straight_func(phi_vals, beta_val, gamma_val, r_max_val, L_val, R_val)
a_straight = a_K_straight_func(phi_vals, beta_val, gamma_val, r_max_val, L_val, R_val)

# Define line styles
line_styles = {
    'inclined': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5},
    'new': {'color': 'blue', 'linestyle': '--', 'linewidth': 2.5},
    'straight': {'color': 'green', 'linestyle': ':', 'linewidth': 2.5}
}

# Function to add parameter text box
def add_param_textbox(ax):
    textstr = f'Parameters:\nR = r_max = {r_max_val} mm\nL = {L_val} mm\nγ = {np.degrees(gamma_val)}°\nβ = {np.degrees(beta_val)}°'
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.annotate(textstr, xy=(0.98, 0.05), xycoords='axes fraction',
              verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Create and save displacement plot
plt.figure(figsize=(12, 6), facecolor='none')
plt.plot(np.degrees(phi_vals), s_inclined, **line_styles['inclined'],
         label='Research (Inclined)')
plt.plot(np.degrees(phi_vals), s_new, **line_styles['new'],
         label='Literature (Inclined)')
plt.plot(np.degrees(phi_vals), s_straight, **line_styles['straight'],
         label='Straight Piston')
plt.xlabel('φ (degrees)', fontsize=12)
plt.ylabel('Displacement (mm)', fontsize=12)
plt.title('Piston Displacement Comparison', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
add_param_textbox(plt.gca())
plt.tight_layout()
plt.savefig('displacement_comparison.png', transparent=True, dpi=300, bbox_inches='tight')

plt.close()

# Create and save velocity plot
plt.figure(figsize=(12, 6), facecolor='none')
plt.plot(np.degrees(phi_vals), v_inclined, **line_styles['inclined'],
         label='Research (Inclined)')
plt.plot(np.degrees(phi_vals), v_new, **line_styles['new'],
         label='Literature (Inclined)')
plt.plot(np.degrees(phi_vals), v_straight, **line_styles['straight'],
         label='Straight Piston')
plt.xlabel('φ (degrees)', fontsize=12)
plt.ylabel('Velocity (mm/rad)', fontsize=12)
plt.title('Piston Velocity Comparison', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
add_param_textbox(plt.gca())
plt.tight_layout()
plt.savefig('velocity_comparison.png', transparent=True, dpi=300, bbox_inches='tight')

plt.close()

# Create and save acceleration plot
plt.figure(figsize=(12, 6), facecolor='none')
plt.plot(np.degrees(phi_vals), a_inclined, **line_styles['inclined'],
         label='Research (Inclined)')
plt.plot(np.degrees(phi_vals), a_new, **line_styles['new'],
         label='Literature (Inclined)')
plt.plot(np.degrees(phi_vals), a_straight, **line_styles['straight'],
         label='Straight Piston')
plt.xlabel('φ (degrees)', fontsize=12)
plt.ylabel('Acceleration (mm/rad²)', fontsize=12)
plt.title('Piston Acceleration Comparison', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
add_param_textbox(plt.gca())
plt.tight_layout()
plt.savefig('acceleration_comparison.png', transparent=True, dpi=300, bbox_inches='tight')

plt.close()

# Create combined plot with all three graphs
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), facecolor='none')

# Displacement plot
ax1.plot(np.degrees(phi_vals), s_inclined, **line_styles['inclined'],
         label='Research (Inclined)')
ax1.plot(np.degrees(phi_vals), s_new, **line_styles['new'],
         label='Literature (Inclined)')
ax1.plot(np.degrees(phi_vals), s_straight, **line_styles['straight'],
         label='Straight Piston')
ax1.set_ylabel('Displacement (mm)', fontsize=12)
ax1.set_title('Piston Displacement Comparison', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Velocity plot
ax2.plot(np.degrees(phi_vals), v_inclined, **line_styles['inclined'],
         label='Research (Inclined)')
ax2.plot(np.degrees(phi_vals), v_new, **line_styles['new'],
         label='Literature (Inclined)')
ax2.plot(np.degrees(phi_vals), v_straight, **line_styles['straight'],
         label='Straight Piston')
ax2.set_ylabel('Velocity (mm/rad)', fontsize=12)
ax2.set_title('Piston Velocity Comparison', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Acceleration plot
ax3.plot(np.degrees(phi_vals), a_inclined, **line_styles['inclined'],
         label='Research (Inclined)')
ax3.plot(np.degrees(phi_vals), a_new, **line_styles['new'],
         label='Literature (Inclined)')
ax3.plot(np.degrees(phi_vals), a_straight, **line_styles['straight'],
         label='Straight Piston')
ax3.set_xlabel('φ (degrees)', fontsize=12)
ax3.set_ylabel('Acceleration (mm/rad²)', fontsize=12)
ax3.set_title('Piston Acceleration Comparison', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Add parameter text box to the combined plot
textstr = f'Parameters:\nR = r_max = {r_max_val} mm\nL = {L_val} mm\nγ = {np.degrees(gamma_val)}°\nβ = {np.degrees(beta_val)}°'
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
plt.annotate(textstr, xy=(0.98, 0.02), xycoords='figure fraction',
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('kinematics_comparison_combined.png', transparent=True, dpi=300, bbox_inches='tight')

plt.close()

print("All comparison plots have been created and saved successfully!")