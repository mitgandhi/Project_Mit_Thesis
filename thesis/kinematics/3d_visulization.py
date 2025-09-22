import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button


def create_frame(ax, origin, R, scale=1.0, label=''):
    """Draw a coordinate frame at given origin with rotation matrix R"""
    colors = ['r', 'g', 'b']
    for i in range(3):
        axis = np.zeros((3, 2))
        axis[:, 0] = origin
        axis[:, 1] = origin + scale * R[:, i]
        ax.plot(axis[0, :], axis[1, :], axis[2, :], colors[i], linewidth=2)
        if label:
            ax.text(axis[1, 1], axis[1, 1], axis[2, 1], f'{label}{["x", "y", "z"][i]}')


def get_normal_vectors(alpha, beta, scale=2.0):
    """Calculate normal vectors n0 and n1"""
    n0 = np.array([0, 1, np.tan(beta)])
    n0 = n0 / np.linalg.norm(n0) * scale

    n1 = np.array([-1, 0, -np.tan(alpha)])
    n1 = n1 / np.linalg.norm(n1) * scale

    return n0, n1


def calculate_L2(L1, gamma, alpha, beta):
    """Calculate L2 based on the equation derived"""
    return L1 / np.cos(gamma)


def transformation_matrices(phi, gamma, L1, L2, R):
    """Create transformation matrices with numerical values"""
    T01 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, L1],
        [0, 0, 0, 1]
    ])

    T12 = np.array([
        [np.cos(phi), -np.sin(phi), 0, 0],
        [np.sin(phi), np.cos(phi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T23 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, R],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T34 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma), 0],
        [0, np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 0, 1]
    ])

    T45 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -L2],
        [0, 0, 0, 1]
    ])

    return T01, T12, T23, T34, T45


def draw_normal_vectors(ax, end_point, alpha, beta):
    """Draw normal vectors and their projections"""
    n0, n1 = get_normal_vectors(alpha, beta)

    ax.quiver(end_point[0], end_point[1], end_point[2],
              n0[0], n0[1], n0[2],
              color='m', label='n0', length=2)

    ax.quiver(end_point[0], end_point[1], end_point[2],
              n1[0], n1[1], n1[2],
              color='c', label='n1', length=2)

    # Draw surface planes
    xx, yy = np.meshgrid([-2, 2], [-2, 2])

    z0 = (-n0[0] * xx - n0[1] * yy) / n0[2] + end_point[2]
    ax.plot_surface(xx + end_point[0], yy + end_point[1], z0,
                    alpha=0.2, color='m')

    z1 = (-n1[0] * xx - n1[1] * yy) / n1[2] + end_point[2]
    ax.plot_surface(xx + end_point[0], yy + end_point[1], z1,
                    alpha=0.2, color='c')


def draw_length_indicators(ax, points, L1, L2):
    """Draw length indicators for L1 and L2"""
    # Draw L1 indicator
    z_mid_L1 = points[1][2] / 2
    ax.plot([0, 0], [0, 0], [0, points[1][2]],
            color='yellow', linestyle='--', linewidth=2, label=f'L1={L1:.2f}')
    ax.text(0, 0.5, z_mid_L1, f'L1={L1:.2f}', color='yellow')

    # Draw L2 indicator
    start_L2 = points[-2]  # Second to last point
    end_L2 = points[-1]  # Last point
    z_mid_L2 = (start_L2[2] + end_L2[2]) / 2
    ax.plot([start_L2[0], end_L2[0]],
            [start_L2[1], end_L2[1]],
            [start_L2[2], end_L2[2]],
            color='#FFA500', linestyle='--', linewidth=2, label=f'L2={L2:.2f}')
    ax.text(start_L2[0], start_L2[1] + 0.5, z_mid_L2, f'L2={L2:.2f}', color='#FFA500')


class AnimationControl:
    def __init__(self, anim):
        self.anim = anim
        self.paused = False

    def toggle_animation(self, event):
        if self.paused:
            self.anim.resume()
        else:
            self.anim.pause()
        self.paused = not self.paused


def update(frame):
    """Update function for animation"""
    ax.cla()

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Parameters with dynamic motion
    L1 = 5.0
    R = 2.0

    # Make phi change with frame but keep gamma constant at -3 degrees
    phi = (frame % 50) * 2 * np.pi / 50  # Complete rotation every 50 frames
    gamma = np.radians(-3)  # Constant -3 degrees converted to radians

    alpha = np.pi / 6
    beta = np.pi / 4

    # Calculate L2 based on the parameters
    L2 = calculate_L2(L1, gamma, alpha, beta)

    # Get transformation matrices
    T01, T12, T23, T34, T45 = transformation_matrices(phi, gamma, L1, L2, R)

    # Calculate cumulative transformations
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45

    # Draw coordinate frames
    create_frame(ax, np.array([0, 0, 0]), np.eye(3), scale=1.0, label='0_')
    create_frame(ax, T01[0:3, 3], T01[0:3, 0:3], scale=1.0, label='1_')
    create_frame(ax, T02[0:3, 3], T02[0:3, 0:3], scale=1.0, label='2_')
    create_frame(ax, T03[0:3, 3], T03[0:3, 0:3], scale=1.0, label='3_')
    create_frame(ax, T04[0:3, 3], T04[0:3, 0:3], scale=1.0, label='4_')
    create_frame(ax, T05[0:3, 3], T05[0:3, 0:3], scale=1.0, label='5_')

    # Draw normal vectors and surfaces
    draw_normal_vectors(ax, T05[0:3, 3], alpha, beta)

    # Get points for drawing links and length indicators
    points = np.array([
        [0, 0, 0],
        T01[0:3, 3],
        T02[0:3, 3],
        T03[0:3, 3],
        T04[0:3, 3],
        T05[0:3, 3]
    ])

    # Draw robot links
    ax.plot(points[:, 0], points[:, 1], points[:, 2],
            color='black', linestyle='--', linewidth=1)

    # Draw length indicators
    draw_length_indicators(ax, points, L1, L2)

    ax.set_title(f'φ={phi:.2f} rad, γ={np.degrees(gamma):.1f}°\nL2={L2:.2f}')
    ax.legend()

    # Add view angle rotation for better visualization
    ax.view_init(elev=20, azim=frame % 360)


# Create figure with space for button
fig = plt.figure(figsize=(12, 13))

# Create main axes for 3D plot
ax = fig.add_subplot(111, projection='3d', position=[0.1, 0.1, 0.8, 0.8])

# Create axes for button
button_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
button = Button(button_ax, 'Play/Pause')

# Create animation
anim = animation.FuncAnimation(fig, update, frames=200, interval=50)

# Create animation control and connect button
animation_control = AnimationControl(anim)
button.on_clicked(animation_control.toggle_animation)

plt.show()