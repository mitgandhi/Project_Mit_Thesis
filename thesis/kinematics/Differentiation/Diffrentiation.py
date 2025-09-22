from sympy import symbols, cos, tan, sin, diff

# Define symbolic variables
theta, R, L, beta, phi = symbols('theta R L beta phi')

# Define displacement equation
disp =   (-R* (tan(beta)/(cos(phi))* ((1- cos(theta)))/(1+ (tan(beta)*tan(phi)*cos(theta)))))
# Calculate velocity (first derivative)
velocity = diff(disp, theta)

# Calculate acceleration (second derivative)
acceleration = diff(velocity, theta)

print("Acceleration equation:")
print(acceleration)

print("Acceleration equation:")
print(velocity)


# Optional: To evaluate with numeric values
def get_acceleration(theta_val, R_val, L_val, beta_val, phi_val):
    return acceleration.subs({
        theta: theta_val,
        R: R_val,
        L: L_val,
        beta: beta_val,
        phi: phi_val
    })

