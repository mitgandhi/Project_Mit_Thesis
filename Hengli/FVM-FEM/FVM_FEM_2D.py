import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator


def fem_2d_heat(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right):
    """
    2D steady-state heat conduction using FEM
    nx, ny: number of elements in x and y directions
    """
    dx = Lx / nx
    dy = Ly / ny

    n_nodes = (nx + 1) * (ny + 1)
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    # Element loop
    for ey in range(ny):
        for ex in range(nx):
            nodes = [
                ey * (nx + 1) + ex,
                ey * (nx + 1) + ex + 1,
                (ey + 1) * (nx + 1) + ex,
                (ey + 1) * (nx + 1) + ex + 1
            ]

            Ke = k / (dx * dy) * np.array([
                [2 * (dx ** 2 + dy ** 2), -2 * dx ** 2, -2 * dy ** 2, 0],
                [-2 * dx ** 2, 2 * (dx ** 2 + dy ** 2), 0, -2 * dy ** 2],
                [-2 * dy ** 2, 0, 2 * (dx ** 2 + dy ** 2), -2 * dx ** 2],
                [0, -2 * dy ** 2, -2 * dx ** 2, 2 * (dx ** 2 + dy ** 2)]
            ]) / (3 * (dx * dy))

            for i in range(4):
                for j in range(4):
                    K[nodes[i], nodes[j]] += Ke[i, j]

    # Apply boundary conditions
    # Bottom
    for i in range(nx + 1):
        idx = i
        K[idx, :] = 0
        K[idx, idx] = 1
        F[idx] = T_bottom

    # Top
    for i in range(nx + 1):
        idx = ny * (nx + 1) + i
        K[idx, :] = 0
        K[idx, idx] = 1
        F[idx] = T_top

    # Left
    for j in range(ny + 1):
        idx = j * (nx + 1)
        K[idx, :] = 0
        K[idx, idx] = 1
        F[idx] = T_left

    # Right
    for j in range(ny + 1):
        idx = j * (nx + 1) + nx
        K[idx, :] = 0
        K[idx, idx] = 1
        F[idx] = T_right

    T = spsolve(csr_matrix(K), F)
    return T.reshape((ny + 1, nx + 1))


def fvm_2d_heat(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right):
    """
    2D steady-state heat conduction using FVM
    nx, ny: number of control volumes
    """
    dx = Lx / nx
    dy = Ly / ny
    n_nodes = nx * ny

    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i

            # Diagonal
            A[idx, idx] = -2 * k * (1 / dx ** 2 + 1 / dy ** 2)

            # x-neighbors
            if i > 0:
                A[idx, idx - 1] = k / dx ** 2
            else:
                b[idx] -= k * T_left / dx ** 2

            if i < nx - 1:
                A[idx, idx + 1] = k / dx ** 2
            else:
                b[idx] -= k * T_right / dx ** 2

            # y-neighbors
            if j > 0:
                A[idx, idx - nx] = k / dy ** 2
            else:
                b[idx] -= k * T_bottom / dy ** 2

            if j < ny - 1:
                A[idx, idx + nx] = k / dy ** 2
            else:
                b[idx] -= k * T_top / dy ** 2

    T_internal = spsolve(csr_matrix(A), b)

    # Include boundary cells
    T = np.zeros((ny + 2, nx + 2))
    T[1:-1, 1:-1] = T_internal.reshape((ny, nx))
    T[0, :] = T_bottom
    T[-1, :] = T_top
    T[:, 0] = T_left
    T[:, -1] = T_right

    return T


# Problem parameters
Lx = Ly = 1.0
k = 1.0
nx = ny = 20

# Boundary conditions
T_bottom = 100
T_top = 0
T_left = 50
T_right = 25

# Solve using both methods
T_fem = fem_2d_heat(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right)
T_fvm = fvm_2d_heat(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right)

# Create grids for plotting
x_fem = np.linspace(0, Lx, nx + 1)
y_fem = np.linspace(0, Ly, ny + 1)
X_fem, Y_fem = np.meshgrid(x_fem, y_fem)

x_fvm = np.linspace(0, Lx, nx + 2)
y_fvm = np.linspace(0, Ly, ny + 2)
X_fvm, Y_fvm = np.meshgrid(x_fvm, y_fvm)

# Interpolate FVM solution to FEM grid for comparison
fvm_interp = RegularGridInterpolator((y_fvm, x_fvm), T_fvm)
points = np.array([(y, x) for y in y_fem for x in x_fem])
T_fvm_on_fem_grid = fvm_interp(points).reshape((ny + 1, nx + 1))

# Calculate difference
diff = np.abs(T_fem - T_fvm_on_fem_grid)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# FEM solution
im1 = ax1.contourf(X_fem, Y_fem, T_fem, levels=20, cmap='hot')
ax1.set_title('FEM Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
plt.colorbar(im1, ax=ax1)

# FVM solution
im2 = ax2.contourf(X_fvm, Y_fvm, T_fvm, levels=20, cmap='hot')
ax2.set_title('FVM Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.colorbar(im2, ax=ax2)

# Difference
im3 = ax3.contourf(X_fem, Y_fem, diff, levels=20, cmap='viridis')
ax3.set_title('Absolute Difference')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.show()

# Print statistics
print("\nSolution Statistics:")
print(f"Maximum Temperature (FEM): {np.max(T_fem):.2f}")
print(f"Maximum Temperature (FVM): {np.max(T_fvm):.2f}")
print(f"Maximum Absolute Difference: {np.max(diff):.2e}")
print(f"Average Absolute Difference: {np.mean(diff):.2e}")


# Add error convergence analysis
def compute_error(nx_test):
    T_fem_test = fem_2d_heat(nx_test, nx_test, Lx, Ly, k, T_bottom, T_top, T_left, T_right)
    T_fvm_test = fvm_2d_heat(nx_test, nx_test, Lx, Ly, k, T_bottom, T_top, T_left, T_right)

    x_fem_test = np.linspace(0, Lx, nx_test + 1)
    y_fem_test = np.linspace(0, Ly, nx_test + 1)
    x_fvm_test = np.linspace(0, Lx, nx_test + 2)
    y_fvm_test = np.linspace(0, Ly, nx_test + 2)

    fvm_interp = RegularGridInterpolator((y_fvm_test, x_fvm_test), T_fvm_test)
    points = np.array([(y, x) for y in y_fem_test for x in x_fem_test])
    T_fvm_interp = fvm_interp(points).reshape((nx_test + 1, nx_test + 1))

    return np.mean(np.abs(T_fem_test - T_fvm_interp))


# Compute errors for different mesh sizes
nx_values = [10, 20, 40, 80]
errors = [compute_error(nx) for nx in nx_values]

# Plot convergence
plt.figure(figsize=(8, 6))
plt.loglog(nx_values, errors, 'bo-', label='Error')
plt.loglog(nx_values, np.array(nx_values) ** (-2), 'r--', label='Second Order Reference')
plt.xlabel('Number of elements per direction')
plt.ylabel('Average absolute difference')
plt.title('Convergence Analysis')
plt.grid(True)
plt.legend()
plt.show()