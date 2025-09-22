import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import time


def chebyshev_points(n):
    """Generate Chebyshev points on [-1,1]"""
    return np.cos(np.pi * np.arange(n - 1, -1, -1) / (n - 1))


def fem_1d(nx, L, k, T_left, T_right):
    """1D heat conduction using Finite Element Method"""
    dx = L / nx

    # Create stiffness matrix and load vector
    K = lil_matrix((nx + 1, nx + 1))
    F = np.zeros(nx + 1)

    # Assembly
    for i in range(nx):
        # Element stiffness matrix
        Ke = k / dx * np.array([[1, -1], [-1, 1]])

        # Assembly to global matrix
        K[i:i + 2, i:i + 2] += Ke

    # Apply boundary conditions
    K[0, :] = 0
    K[0, 0] = 1
    K[-1, :] = 0
    K[-1, -1] = 1
    F[0] = T_left
    F[-1] = T_right

    # Solve system
    T = spsolve(csr_matrix(K), F)
    return T


def fvm_1d(nx, L, k, T_left, T_right):
    """1D heat conduction using Finite Volume Method"""
    dx = L / nx

    # Create system matrix
    A = lil_matrix((nx, nx))
    b = np.zeros(nx)

    # Interior points
    for i in range(nx):
        # Diagonal term
        A[i, i] = -2 * k / dx ** 2

        # Off-diagonal terms
        if i > 0:
            A[i, i - 1] = k / dx ** 2
        else:
            b[i] -= k * T_left / dx ** 2

        if i < nx - 1:
            A[i, i + 1] = k / dx ** 2
        else:
            b[i] -= k * T_right / dx ** 2

    # Solve system
    T_internal = spsolve(csr_matrix(A), b)

    # Add boundary points
    T = np.zeros(nx + 2)
    T[1:-1] = T_internal
    T[0] = T_left
    T[-1] = T_right

    return T


def spectral_1d(nx, L, k, T_left, T_right):
    """1D heat conduction using Spectral Method"""
    # Generate Chebyshev points
    x = chebyshev_points(nx)
    x = (x + 1) * L / 2  # Scale to [0,L]

    # Create differentiation matrix
    D = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(nx):
            if i != j:
                D[i, j] = (-1) ** (i + j) / (x[i] - x[j])
        D[i, i] = -sum(D[i, :])

    # Second derivative matrix
    D2 = D @ D

    # Create system matrix and right-hand side
    A = k * D2
    b = np.zeros(nx)

    # Apply boundary conditions
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1
    b[0] = T_left
    b[-1] = T_right

    # Solve system
    T = np.linalg.solve(A, b)
    return T, x


def fdm_1d(nx, L, k, T_left, T_right):
    """1D heat conduction using Finite Difference Method"""
    dx = L / (nx - 1)

    # Initialize temperature array
    T = np.zeros(nx)
    T[0] = T_left
    T[-1] = T_right

    # Create system matrix
    A = np.zeros((nx - 2, nx - 2))
    b = np.zeros(nx - 2)

    # Fill matrix using central difference
    for i in range(nx - 2):
        if i > 0:
            A[i, i - 1] = k / dx ** 2
        A[i, i] = -2 * k / dx ** 2
        if i < nx - 3:
            A[i, i + 1] = k / dx ** 2

    # Modify RHS for boundary conditions
    b[0] = -k * T_left / dx ** 2
    b[-1] = -k * T_right / dx ** 2

    # Solve system
    T[1:-1] = np.linalg.solve(A, b)

    return T


# Problem parameters
L = 1.0  # Length of domain
k = 1.0  # Thermal conductivity
nx = 50  # Number of points
T_left = 100  # Left boundary temperature
T_right = 0  # Right boundary temperature

# Create x-coordinates for plotting
x_uniform = np.linspace(0, L, nx)

# Solve using all methods and measure time
methods = ['FEM', 'FVM', 'Spectral', 'FDM']
times = []
solutions = []
x_coords = []

# FEM solution
start_time = time.time()
T_fem = fem_1d(nx - 1, L, k, T_left, T_right)
times.append(time.time() - start_time)
solutions.append(T_fem)
x_coords.append(np.linspace(0, L, nx))

# FVM solution
start_time = time.time()
T_fvm = fvm_1d(nx - 1, L, k, T_left, T_right)
times.append(time.time() - start_time)
solutions.append(T_fvm)
x_coords.append(np.linspace(0, L, nx + 1))

# Spectral solution
start_time = time.time()
T_spec, x_spec = spectral_1d(nx, L, k, T_left, T_right)
times.append(time.time() - start_time)
solutions.append(T_spec)
x_coords.append(x_spec)

# FDM solution
start_time = time.time()
T_fdm = fdm_1d(nx, L, k, T_left, T_right)
times.append(time.time() - start_time)
solutions.append(T_fdm)
x_coords.append(x_uniform)

# Exact solution
x_exact = np.linspace(0, L, 200)
T_exact = T_left + (T_right - T_left) * x_exact / L

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(x_exact, T_exact, 'k-', label='Exact', linewidth=2)
markers = ['o', 's', '^', 'v']
for i, (method, solution, x, marker) in enumerate(zip(methods, solutions, x_coords, markers)):
    plt.plot(x, solution, marker, label=f'{method} (t={times[i]:.3f}s)', markersize=6)

plt.xlabel('Position (x)')
plt.ylabel('Temperature (T)')
plt.title('1D Heat Conduction - Comparison of Numerical Methods')
plt.grid(True)
plt.legend()
plt.show()

# Error analysis
print("\nMethod Comparison:")
print("------------------")
for method, solution, x, t in zip(methods, solutions, x_coords, times):
    # Interpolate exact solution to method's grid points
    T_exact_interp = T_left + (T_right - T_left) * x / L
    error = np.max(np.abs(solution - T_exact_interp))

    print(f"\n{method}:")
    print(f"Maximum Temperature: {np.max(solution):.2f}")
    print(f"Minimum Temperature: {np.min(solution):.2f}")
    print(f"Maximum Error: {error:.2e}")
    print(f"Computation Time: {t:.3f} seconds")
    print(f"Number of nodes: {len(solution)}")


# Convergence analysis
def compute_errors(n_points):
    errors = []
    for method in [fem_1d, fvm_1d, spectral_1d, fdm_1d]:
        if method == spectral_1d:
            T, x = method(n_points, L, k, T_left, T_right)
        else:
            T = method(n_points - 1, L, k, T_left, T_right)
            x = np.linspace(0, L, len(T))
        T_exact = T_left + (T_right - T_left) * x / L
        error = np.max(np.abs(T - T_exact))
        errors.append(error)
    return errors


n_values = [10, 20, 40, 80]
all_errors = [compute_errors(n) for n in n_values]

plt.figure(figsize=(10, 6))
for i, method in enumerate(methods):
    errors = [err[i] for err in all_errors]
    plt.loglog(n_values, errors, 'o-', label=method)

plt.loglog(n_values, np.array(n_values) ** (-2), 'k--', label='Second Order Reference')
plt.xlabel('Number of points')
plt.ylabel('Maximum Error')
plt.title('Convergence Analysis')
plt.grid(True)
plt.legend()
plt.show()