import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import time


def chebyshev_points(n):
    """Generate Chebyshev points on [-1,1]"""
    return np.cos(np.pi * np.arange(n - 1, -1, -1) / (n - 1))


def fem_2d(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right):
    """2D heat conduction using Finite Element Method"""
    dx = Lx / nx
    dy = Ly / ny

    n_nodes = (nx + 1) * (ny + 1)
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    # Assembly
    for ey in range(ny):
        for ex in range(nx):
            nodes = [
                ey * (nx + 1) + ex,
                ey * (nx + 1) + ex + 1,
                (ey + 1) * (nx + 1) + ex,
                (ey + 1) * (nx + 1) + ex + 1
            ]

            # Element stiffness matrix
            Ke = k / (dx * dy) * np.array([
                [2 * (dx ** 2 + dy ** 2), -2 * dx ** 2, -2 * dy ** 2, 0],
                [-2 * dx ** 2, 2 * (dx ** 2 + dy ** 2), 0, -2 * dy ** 2],
                [-2 * dy ** 2, 0, 2 * (dx ** 2 + dy ** 2), -2 * dx ** 2],
                [0, -2 * dy ** 2, -2 * dx ** 2, 2 * (dx ** 2 + dy ** 2)]
            ]) / (3 * (dx * dy))

            # Assembly
            for i in range(4):
                for j in range(4):
                    K[nodes[i], nodes[j]] += Ke[i, j]

    # Apply boundary conditions
    for i in range(nx + 1):
        # Bottom
        idx = i
        K[idx, :] = 0
        K[idx, idx] = 1
        F[idx] = T_bottom

        # Top
        idx = ny * (nx + 1) + i
        K[idx, :] = 0
        K[idx, idx] = 1
        F[idx] = T_top

    for j in range(ny + 1):
        # Left
        idx = j * (nx + 1)
        K[idx, :] = 0
        K[idx, idx] = 1
        F[idx] = T_left

        # Right
        idx = j * (nx + 1) + nx
        K[idx, :] = 0
        K[idx, idx] = 1
        F[idx] = T_right

    T = spsolve(csr_matrix(K), F)
    return T.reshape((ny + 1, nx + 1))


def fvm_2d(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right):
    """2D heat conduction using Finite Volume Method"""
    dx = Lx / nx
    dy = Ly / ny

    n_nodes = nx * ny
    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i

            # Central coefficient
            A[idx, idx] = -2 * k * (1 / dx ** 2 + 1 / dy ** 2)

            # Neighbors
            if i > 0:
                A[idx, idx - 1] = k / dx ** 2
            if i < nx - 1:
                A[idx, idx + 1] = k / dx ** 2
            if j > 0:
                A[idx, idx - nx] = k / dy ** 2
            if j < ny - 1:
                A[idx, idx + nx] = k / dy ** 2

            # Boundary conditions
            if i == 0: b[idx] -= k * T_left / dx ** 2
            if i == nx - 1: b[idx] -= k * T_right / dx ** 2
            if j == 0: b[idx] -= k * T_bottom / dy ** 2
            if j == ny - 1: b[idx] -= k * T_top / dy ** 2

    T_internal = spsolve(csr_matrix(A), b)
    T = np.zeros((ny + 2, nx + 2))
    T[1:-1, 1:-1] = T_internal.reshape((ny, nx))
    T[0, :] = T_bottom
    T[-1, :] = T_top
    T[:, 0] = T_left
    T[:, -1] = T_right

    return T


def spectral_2d(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right):
    """2D heat conduction using Spectral Method"""
    x = chebyshev_points(nx)
    y = chebyshev_points(ny)

    x = (x + 1) * Lx / 2
    y = (y + 1) * Ly / 2

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    n = nx * ny
    A = lil_matrix((n, n))
    b = np.zeros(n)

    # Interior points
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            idx = j * nx + i
            w = 2 / (x[i + 1] - x[i - 1])
            A[idx, idx] = -2 * k * (w / dx + w / dy)
            A[idx, idx - 1] = k * w / dx
            A[idx, idx + 1] = k * w / dx
            A[idx, idx - nx] = k * w / dy
            A[idx, idx + nx] = k * w / dy

    # Boundary conditions
    for i in range(nx):
        idx = i
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = T_bottom

        idx = (ny - 1) * nx + i
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = T_top

    for j in range(ny):
        idx = j * nx
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = T_left

        idx = j * nx + nx - 1
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = T_right

    T = spsolve(csr_matrix(A), b)
    return T.reshape((ny, nx)), x, y


def fdm_2d(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right):
    """2D heat conduction using Finite Difference Method"""
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    T = np.zeros((ny, nx))
    T[0, :] = T_bottom
    T[-1, :] = T_top
    T[:, 0] = T_left
    T[:, -1] = T_right

    cx = k / dx ** 2
    cy = k / dy ** 2
    c = -2 * (cx + cy)

    max_iter = 1000
    tolerance = 1e-6

    for iter in range(max_iter):
        T_old = T.copy()

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                T[i, j] = -(cx * (T[i, j - 1] + T[i, j + 1]) +
                            cy * (T[i - 1, j] + T[i + 1, j])) / c

        if np.max(np.abs(T - T_old)) < tolerance:
            break

    return T


# Problem setup
Lx = Ly = 1.0
k = 1.0
nx = ny = 21

# Boundary conditions
T_bottom = 100
T_top = 0
T_left = 50
T_right = 25

# Solve using all methods and measure time
methods = ['FEM', 'FVM', 'Spectral', 'FDM']
times = []
solutions = []

start_time = time.time()
T_fem = fem_2d(nx - 1, ny - 1, Lx, Ly, k, T_bottom, T_top, T_left, T_right)
times.append(time.time() - start_time)
solutions.append(T_fem)

start_time = time.time()
T_fvm = fvm_2d(nx - 1, ny - 1, Lx, Ly, k, T_bottom, T_top, T_left, T_right)
times.append(time.time() - start_time)
solutions.append(T_fvm)

start_time = time.time()
T_spec, x_spec, y_spec = spectral_2d(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right)
times.append(time.time() - start_time)
solutions.append(T_spec)

start_time = time.time()
T_fdm = fdm_2d(nx, ny, Lx, Ly, k, T_bottom, T_top, T_left, T_right)
times.append(time.time() - start_time)
solutions.append(T_fdm)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()

for idx, (method, solution, t) in enumerate(zip(methods, solutions, times)):
    x = np.linspace(0, Lx, solution.shape[1])
    y = np.linspace(0, Ly, solution.shape[0])
    X, Y = np.meshgrid(x, y)

    im = axs[idx].contourf(X, Y, solution, levels=20, cmap='hot')
    axs[idx].set_title(f'{method} (Time: {t:.3f}s)')
    axs[idx].set_xlabel('x')
    axs[idx].set_ylabel('y')
    plt.colorbar(im, ax=axs[idx])

plt.tight_layout()
plt.show()

# Print statistics
print("\nMethod Comparison:")
print("------------------")
for method, solution, t in zip(methods, solutions, times):
    print(f"\n{method}:")
    print(f"Maximum Temperature: {np.max(solution):.2f}")
    print(f"Minimum Temperature: {np.min(solution):.2f}")
    print(f"Computation Time: {t:.3f} seconds")
    print(f"Number of nodes: {solution.size}")

# Compute differences between methods
print("\nSolution Differences (RMS):")
print("--------------------------")
for i, method1 in enumerate(methods):
    for j, method2 in enumerate(methods[i + 1:], i + 1):
        diff = np.sqrt(np.mean((solutions[i][1:-1, 1:-1] - solutions[j][1:-1, 1:-1]) ** 2))
        print(f"{method1} vs {method2}: {diff:.2e}")