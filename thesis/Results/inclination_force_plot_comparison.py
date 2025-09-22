import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams.update({
    # ---- Fonts ----
    "font.family": "serif",          # safe default shipped with Matplotlib is "DejaVu Serif"
    "font.size": 14,                 # base font size
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    # ---- Lines / markers ----
    "lines.linewidth": 2.5,
    "lines.markersize": 6,

    # ---- Figure / axes ----
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linewidth": 1.0,
    "grid.color": "0.7",

    # ---- Legend ----
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.fancybox": True,

    # ---- Savefig ----
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": True,

    # ---- Mathtext to match serif ----
    "mathtext.fontset": "dejavuserif",
})
def calculate_piston_dynamics(R, r, r_inner, Alpha, gamma_values, N, mass, pdc_data, phi):
    """
    Calculates various forces, accelerations, velocities, and displacements for a piston mechanism.
    """
    # Store input parameters
    results = {
        'input_parameters': {
            'R': R, 'r': r, 'r_inner': r_inner, 'Alpha': Alpha,
            'gamma_values': gamma_values, 'N': N, 'mass': mass
        },
        'phi': phi,
        'pdc_data': pdc_data
    }

    # Basic calculations
    pCase = 1.0  # Case pressure [bar]
    area_k = np.pi * (r ** 2 - r_inner ** 2)  # Piston cross-sectional area [m²]
    pressure_force = (pdc_data - pCase) * area_k  # Pressure force [N]

    # Convert angles and calculate angular velocity
    beta_rad = np.radians(Alpha)  # Swashplate angle [rad]
    omega = (N * np.pi / 30)  # Angular velocity [rad/s]
    phi_rad = np.radians(phi)  # Shaft angles [rad]

    # Calculate time step for integration
    if len(phi) > 1:
        phi_rad_diff = np.radians(phi[1] - phi[0])
        dt = phi_rad_diff / omega
    else:
        dt = 1.0 / N  # Fallback

    # Store common results
    results['pressure_force'] = pressure_force
    results['area_k'] = area_k
    results['omega'] = omega
    results['beta_rad'] = beta_rad
    results['dt'] = dt

    # Initialize gamma-dependent results
    gamma_results = {}

    for g in gamma_values:
        gamma_results[g] = {}
        zeta_rad = np.radians(g)  # Current gamma angle [rad]

        # Calculate variable radius
        r_temp = R - (2 * R * np.tan(beta_rad) *np.tan(zeta_rad)  * (1 - np.cos(phi_rad)))

        # Inertial forces
        FaKz = mass * (omega ** 2) * r_temp * (np.tan(beta_rad) / np.cos(zeta_rad)) * np.cos(phi_rad)
        acceleration = FaKz / mass

        # Centrifugal forces
        FwK_inclined = mass * (omega ** 2) * r_temp
        Fwkz = FwK_inclined * np.sin(zeta_rad)
        Fwky = FwK_inclined * np.cos(zeta_rad)

        # Forces on piston in inclined plane
        FTKy = 0.0  # Friction force (set to zero)
        FAK_inclined = pressure_force + FaKz + FTKy

        # Transform to global frame
        FAKz = FAK_inclined * np.cos(zeta_rad)
        FAKy = FAK_inclined * np.sin(zeta_rad)

        # Reaction forces
        FSK = FAKz / np.cos(beta_rad)
        FSKy = FSK * np.sin(beta_rad)
        FSKx = -FSK * np.sin(zeta_rad)

        # Total lateral force
        QuerKraft = FAKy + FSKy

        # Numerical integration for velocity and displacement
        velocity = np.zeros_like(acceleration)
        displacement = np.zeros_like(acceleration)

        for i in range(1, len(acceleration)):
            velocity[i] = velocity[i - 1] + acceleration[i] * dt
            displacement[i] = displacement[i - 1] + velocity[i] * dt

        # Store all results for this gamma
        gamma_results[g] = {
            'r_temp': r_temp,
            'acceleration': acceleration,
            'inertial_force': FaKz,
            'centrifugal_force_inclined': FwK_inclined,
            'centrifugal_force_z': Fwkz,
            'centrifugal_force_y': Fwky,
            'force_inclined': FAK_inclined,
            'FAKz': FAKz,
            'FAKy': FAKy,
            'FSK': FSK,
            'FSKy': FSKy,
            'FSKx': FSKx,
            'QuerKraft': QuerKraft,
            'velocity': velocity,
            'displacement': displacement,
            'FDK': pressure_force
        }

    results['gamma_results'] = gamma_results
    return results


def load_pdc_data(file_path):
    """Load PDC data from CSV file"""
    try:
        # Try different separators
        try:
            pdc_df = pd.read_csv(file_path, sep='\t', comment='%')
        except:
            try:
                pdc_df = pd.read_csv(file_path, sep=r'\s+', comment='%')
            except:
                pdc_df = pd.read_csv(file_path, comment='%')

        # Check for PDC column
        if 'PDC' not in pdc_df.columns:
            if 'pDC' in pdc_df.columns:
                pdc_df.rename(columns={'pDC': 'PDC'}, inplace=True)
            else:
                print(f"Available columns: {list(pdc_df.columns)}")
                raise ValueError("CSV file must have a column named 'PDC' or 'pDC'.")

        pdc_data = pdc_df['PDC'].to_numpy()

        if pdc_data.ndim == 0 or len(pdc_data) == 0:
            raise ValueError("PDC data is invalid.")

        return pdc_data

    except Exception as e:
        print(f"Error loading PDC data: {e}")
        return None


def create_output_csv(results, gamma_value=0, output_filename="python_forces_gamma_0.csv"):
    """
    Create CSV with extended columns for a given gamma value
    """
    gamma_data = results['gamma_results'][gamma_value]
    phi = results['phi']
    omega = results['omega']
    dt = results['dt']

    # Time and revolution
    time_array = np.arange(len(phi)) * dt
    revolution = phi / 360  # revolution fraction

    # FKx, FKy, FK
    FKx = gamma_data['FSKx']
    FKy = gamma_data['FSKy'] + gamma_data['FAKy']
    FK = np.sqrt(FKx ** 2 + FKy ** 2)

    # FaK
    FaK = gamma_data['inertial_force']

    # FDK
    FDK = gamma_data['FDK']

    # FwK = centrifugal force in inclined direction
    FwK = gamma_data['centrifugal_force_inclined']

    # DataFrame with required columns
    df = pd.DataFrame({
        '%time': time_array,
        'revolution': revolution,
        'shaft_angle': phi,
        'FAKz': gamma_data['FAKz'],
        'FSKx': gamma_data['FSKx'],
        'FSKy': gamma_data['FSKy'],
        'FSKz': gamma_data['FSK'],
        'FKx': FKx,
        'FKy': FKy,
        'FK': FK,
        'FaK': FaK,
        'FDK': FDK,
        'FwK': FwK,
        'FAKy': gamma_data['FAKy']
    })

    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"CSV file saved as: {output_filename}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == "__main__":
    # Input parameters
    R = 42.08 * 0.001  # meters
    r = 9.73205 * 0.001  # meters
    r_inner = 1.5 * 0.001  # meters
    Alpha = 14  # degrees
    gamma_values = [0,2,4,6,5, 8,10]  # degrees
    N = 4400  # RPM,
    mass = 0.103117 # kg

    # File path
    pdc_file_path = "P_DC_subset.csv"

    print("Loading PDC data...")
    pdc_data = load_pdc_data(pdc_file_path)

    if pdc_data is not None:
        # Generate shaft angles
        phi = np.linspace(0, 360, len(pdc_data), endpoint=False)

        print("Running piston dynamics calculation...")
        results = calculate_piston_dynamics(R, r, r_inner, Alpha, gamma_values, N, mass, pdc_data, phi)

        # Create CSV outputs for all gamma values
        for gamma in gamma_values:
            output_file = f"python_forces_gamma_{gamma}.csv"
            print(f"\nCreating CSV output for gamma = {gamma} degrees...")
            create_output_csv(results, gamma_value=gamma, output_filename=output_file)

    else:
        print("Failed to load PDC data. Please check the file path and format.")