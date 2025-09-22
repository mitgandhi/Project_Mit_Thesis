import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
radius = 1
num_points = 360
theta = np.linspace(0, 2 * np.pi, num_points)
degrees = np.linspace(0, 360, num_points)  # For x-axis of line plot (0 to 360)

# Circle coordinates (counter-clockwise rotation)
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)

# Line coordinates (unwrapped circle - left to right)
x_line = degrees
y_line = np.zeros_like(degrees)

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Setup Circle plot
ax1.set_aspect('equal')
ax1.set_xlim(-radius * 1.2, radius * 1.2)
ax1.set_ylim(-radius * 1.2, radius * 1.2)
ax1.set_title("Circle")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.grid(True)
ax1.plot(x_circle, y_circle, color='lightgray')
circle_dot, = ax1.plot([], [], 'ro')

# Setup Line plot with proper ticks
ax2.set_xlim(0, 360)
ax2.set_ylim(-0.5, 0.5)
ax2.set_title("Unwrapped Line")
ax2.set_xlabel("Angle (degrees)")
ax2.set_xticks([0, 90, 180, 270, 360])
ax2.set_yticks([])
ax2.grid(True)
ax2.plot(x_line, y_line, color='lightgray')
line_dot, = ax2.plot([], [], 'bo')
line_trail, = ax2.plot([], [], 'b-', linewidth=2)


# Update function - ensure line moves left to right (0 to 360)
def update(i):
    # Circle moves counter-clockwise
    circle_dot.set_data([x_circle[i]], [y_circle[i]])

    # Line moves left to right (0 to 360)
    line_dot.set_data([degrees[i]], [0])

    # Create the trail from 0 to current position
    line_trail.set_data(degrees[:i + 1], y_line[:i + 1])

    return circle_dot, line_dot, line_trail


# Create animation
ani = FuncAnimation(fig, update, frames=num_points, interval=20, blit=True)

# Save as GIF
gif_path = "circle_unwrap_0_to_360.gif"
ani.save(gif_path, writer=PillowWriter(fps=30))
print(f"GIF saved as: {gif_path}")