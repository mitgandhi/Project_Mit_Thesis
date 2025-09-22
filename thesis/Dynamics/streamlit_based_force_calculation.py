import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.figsize': (10, 6),
'font.family': 'serif'
})

# === Main Calculation Function ===
def calculate_piston_dynamics(R, r, r_inner, Alpha, gamma_values, N, mass, pdc_data, phi):
    """
    Calculates various forces and accelerations for a piston mechanism
    based on given geometric, operational, and pressure data,
    incorporating formulas from the provided C++ snippet.

    Args:
        R (float): Radius of the main shaft in meters.
        r (float): Outer radius of the piston in meters.
        r_inner (float): Inner radius of the piston in meters.
        Alpha (float): Swashplate angle in degrees.
        gamma_values (list): List of gamma angles in degrees.
        N (float): Rotational speed in RPM.
        mass (float): Mass of the piston in kg.
        pdc_data (np.array): Pressure data (PDC) over the shaft angle.
        phi (np.array): Shaft angles in degrees corresponding to pdc_data.

    Returns:
        dict: A dictionary containing all input parameters and calculated
              force/acceleration data, organized by gamma value where applicable.
    """
    data = {
        'R': R, 'r': r, 'r_inner': r_inner, 'Alpha': Alpha, 'gamma': gamma_values,
        'N': N, 'mass': mass, 'PDC': pdc_data, 'phi': phi
    }

    pCase = 1.0  # Assumed case pressure, corresponds to pCase in C++ snippet
    area_k = np.pi * (r ** 2 - r_inner ** 2) # Piston cross-sectional area, corresponds to AreaK in C++ snippet
    # Pressure force (FDK in C++ snippet)
    data['Pressure_Force'] = (pdc_data - pCase) * area_k

    beta_rad = np.radians(Alpha) # Convert Alpha to radians, corresponds to beta in C++ snippet
    omega = (N * np.pi / 30) # Angular velocity in rad/s, corresponds to omega in C++ snippet
    phi_rad = np.radians(phi) # Convert shaft angles to radians, corresponds to phi in C++ snippet

    # Initialize dictionaries to store gamma-dependent results
    data['Acceleration'] = {}
    data['Inertial_Force'] = {}
    data['FAKz'] = {}
    data['FAKy'] = {}
    data['FSK'] = {}
    data['FSKy'] = {}
    data['FSKx'] = {}
    data['QuerKraft'] = {}

    for g in gamma_values:
        zeta_rad = np.radians(g) # Convert current gamma to radians, corresponds to zeta in C++ snippet

        # Calculate r_temp as in C++ snippet (using R for rB)
        # double r_temp= rB - (2*rB*tan(beta)*tan(zeta)*(1-cos(phi)));
        r_temp = R - (2 * R * np.tan(beta_rad) * np.tan(zeta_rad) * (1 - np.cos(phi_rad)))

        # Center of mass inertial force z directions (FaKz in C++ snippet)
        # FaKz = mK * (omega*omega) * r_temp * (tan(beta)/cos(zeta))*cos(phi);
        FaKz_cxx = mass * (omega**2) * r_temp * (np.tan(beta_rad) / np.cos(zeta_rad)) * np.cos(phi_rad)
        data['Inertial_Force'][g] = FaKz_cxx # This is the inertial force component along the piston axis
        data['Acceleration'][g] = FaKz_cxx / mass # Derive acceleration from the calculated inertial force

        # Centrifugal force (FwK_inclined in C++ snippet)
        # FwK_inclined = mK * (omega*omega) * r_temp;
        FwK_inclined_cxx = mass * (omega**2) * r_temp

        # Centrifugal components for transforming inclined piston frame to global frame
        # Fwkz = FwK_inclined * sin(zeta);
        Fwkz_cxx = FwK_inclined_cxx * np.sin(zeta_rad)
        # Fwky = FwK_inclined * cos(zeta);
        Fwky_cxx = FwK_inclined_cxx * np.cos(zeta_rad)

        # Friction forces are set to zero as per C++ snippet
        FTKy = 0.0
        FTGx = 0.0
        FTGy = 0.0

        # Forces on the piston acting in inclined plane (FAK_inclined in C++ snippet)
        # FAK_inclined = FDK + FaKz + FTKy;
        FAK_inclined_cxx = data['Pressure_Force'] + FaKz_cxx + FTKy

        # From inclined plane frame to global frame
        # FAKz = FAK_inclined * cos(zeta);
        data['FAKz'][g] = FAK_inclined_cxx * np.cos(zeta_rad)
        # FAKy = FAK_inclined * sin(zeta);
        data['FAKy'][g] = FAK_inclined_cxx * np.sin(zeta_rad)

        # Reaction forces for inclined (FSK in C++ snippet)
        # FSK = FAKz / cos(beta);
        data['FSK'][g] = data['FAKz'][g] / np.cos(beta_rad)

        # Radial Component of FSK (FSKy in C++ snippet)
        # FSKy = FSK * sin(beta);
        data['FSKy'][g] = data['FSK'][g] * np.sin(beta_rad)

        # X-component of FSK (FSKx in C++ snippet, derived from context)
        # FSKx = - FSK * sin(zeta);
        data['FSKx'][g] = -data['FSK'][g] * np.sin(zeta_rad)

        # Total lateral force (QuerKraft in original Python, corresponds to (FSKy + FAKy) in C++ context)
        data['QuerKraft'][g] = data['FAKy'][g] + data['FSKy'][g]

        # Note: FKx, FKy, MKx, MKy, FK, MK from the C++ snippet are not calculated here
        # because they depend on lSK and zRK, which are not provided in the Python static inputs.

    return data


# === Export to ZIP ===
def export_results_to_zip(results, output_path="piston_dynamics_results.zip"):
    """
    Exports the calculated results to a ZIP file containing CSVs for parameters,
    individual gamma data, and consolidated data.

    Args:
        results (dict): The dictionary containing all calculated data.
        output_path (str): The desired name for the output ZIP file.
    """
    with zipfile.ZipFile(output_path, 'w') as zf:
        # Save parameters
        params_df = pd.DataFrame({
            'Parameter': ['R (m)', 'r (m)', 'r_inner (m)', 'Alpha (degrees)', 'N (RPM)', 'mass (kg)'],
            'Value': [results['R'], results['r'], results['r_inner'],
                      results['Alpha'], results['N'], results['mass']]
        })
        zf.writestr('parameters.csv', params_df.to_csv(index=False))

        # Save data for each gamma value
        for g in results['gamma']:
            gamma_data = {'phi': results['phi']}

            for key in ['Acceleration', 'Inertial_Force', 'FAKz', 'FAKy',
                        'FSK', 'FSKy', 'FSKx', 'QuerKraft']:
                gamma_data[key] = results[key][g]

            gamma_data['Pressure_Force'] = results['Pressure_Force'] # Pressure force is common
            df = pd.DataFrame(gamma_data)
            zf.writestr(f'gamma_{g}_data.csv', df.to_csv(index=False))

        # Save consolidated data for all gamma values
        consolidated = {'phi': results['phi'], 'Pressure_Force': results['Pressure_Force']}
        for key in ['Acceleration', 'Inertial_Force', 'FAKz', 'FAKy', 'FSK', 'FSKy', 'FSKx', 'QuerKraft']:
            for g in results['gamma']:
                consolidated[f'{key}_gamma_{g}'] = results[key][g]

        df_all = pd.DataFrame(consolidated)
        zf.writestr('all_gamma_data.csv', df_all.to_csv(index=False))

    print(f"✅ Data exported to {output_path}")

# === Plotting Functions ===
def plot_series(results, key, ylabel):
    """
    Plots a specific series (e.g., FAKz) for all gamma values.

    Args:
        results (dict): The dictionary containing all calculated data.
        key (str): The key for the data series to plot (e.g., 'FAKz').
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    for g in results['gamma']:
        plt.plot(results['phi'], results[key][g], label=f'γ={g}°')
    plt.xlabel("Shaft Angle (deg)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 360) # Explicitly set x-axis limits
    plt.tight_layout()
    plt.savefig(f'{key}_vs_shaft_angle.png')

def plot_all_forces_at_gamma_zero(results):
    """
    Plots all force components for gamma = 0 degrees on a single plot.

    Args:
        results (dict): The dictionary containing all calculated data.
    """
    gamma_zero = 0 # Target gamma value

    # Check if gamma_zero is in the calculated gamma values
    if gamma_zero not in results['gamma']:
        print(f"Error: Gamma value {gamma_zero} not found in calculated results.")
        return

    plt.figure(figsize=(10, 8))
    plt.plot(results['phi'], results['Pressure_Force'], label='FDK (N)', linestyle='--')
    plt.plot(results['phi'], results['Inertial_Force'][gamma_zero], label=f'FaK (N)')
    plt.plot(results['phi'], results['FAKz'][gamma_zero], label=f'FAKz (N)')

    plt.plot(results['phi'], results['FSK'][gamma_zero], label=f'FSK (N)')
    plt.plot(results['phi'], results['FSKy'][gamma_zero], label=f'FSKy (N)')



    plt.xlabel("Shaft Angle (deg)")
    plt.ylabel("Force (N)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
    plt.grid(True)
    plt.xlim(0, 360) # Explicitly set x-axis limits

    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True,
              fancybox=True, shadow=False, framealpha=0.9, edgecolor='black')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig('all_forces_gamma_zero.png', transparent=True, facecolor='none', edgecolor='none')


def plot_pressure_force(results):
    """Plot DK - Pressure Force"""
    plt.figure(figsize=(8, 5))
    plt.plot(results['phi'], results['Pressure_Force'], label='FDK - Pressure Force')
    plt.xlabel("Shaft Angle (°)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 360) # Explicitly set x-axis limits
    plt.ylim(0, 10000)
    plt.tight_layout()
    plt.savefig('pressure_force.png')

def plot_total_piston_force(results, gamma):
    """Plot AK - Total Piston Force (FAK_inclined) for given gamma"""
    if gamma not in results['gamma']:
        print(f"Gamma = {gamma}° not found.")
        return
    FAK_inclined = results['FAKz'][gamma] / np.cos(np.radians(results['Alpha']))  # Reverse projection
    plt.figure(figsize=(8, 5))
    plt.plot(results['phi'], FAK_inclined, label=f'AK - Total Piston Force')
    plt.xlabel("Shaft Angle (°)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 360) # Explicitly set x-axis limits
    plt.ylim(0, 10000)
    plt.tight_layout()
    plt.savefig(f'total_piston_force_gamma_{gamma}.png')

def plot_reaction_force(results, gamma):
    """Plot FSK - Reaction Force for given gamma"""
    if gamma not in results['gamma']:
        print(f"Gamma = {gamma}° not found.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(results['phi'], results['FSK'][gamma], label=f'FSK - Reaction Force')
    plt.xlabel("Shaft Angle (°)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 360) # Explicitly set x-axis limits
    plt.tight_layout()
    plt.savefig(f'reaction_force_gamma_{gamma}.png')

def plot_centrifugal_force(results, gamma):
    """Plot FwK - Centrifugal Force for given gamma"""
    if gamma not in results['gamma']:
        print(f"Gamma = {gamma}° not found.")
        return
    beta_rad = np.radians(results['Alpha'])
    omega = results['N'] * np.pi / 30
    phi_rad = np.radians(results['phi'])
    R = results['R']
    zeta_rad = np.radians(gamma)
    r_temp = R - (2 * R * np.tan(beta_rad) * np.tan(zeta_rad) * (1 - np.cos(phi_rad)))
    FwK = results['mass'] * omega**2 * r_temp

    plt.figure(figsize=(8, 5))
    plt.plot(results['phi'], FwK, label=f'FwK - Centrifugal Force')
    plt.xlabel("Shaft Angle (°)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 360) # Explicitly set x-axis limits
    plt.tight_layout()
    plt.savefig(f'centrifugal_force_gamma_{gamma}.png')

def plot_inertial_force(results, gamma):
    """Plot FaK - Inertial Force for given gamma"""
    if gamma not in results['gamma']:
        print(f"Gamma = {gamma}° not found.")
        return
    plt.figure(figsize=(8, ))
    plt.plot(results['phi'], results['Inertial_Force'][gamma], label=f'FaK - Inertial Force')
    plt.xlabel("Shaft Angle (°)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 360) # Explicitly set x-axis limits
    plt.tight_layout()
    plt.savefig(f'inertial_force_gamma_{gamma}.png')

def plot_centrifugal_and_inertial_forces(results, gamma):
    """
    Plots Centrifugal Force and Inertial Force for a given gamma on a single plot,
    formatted for LaTeX, without a title.
    """
    if gamma not in results['gamma']:
        print(f"Error: Gamma value {gamma} not found in calculated results.")
        return

    beta_rad = np.radians(results['Alpha'])
    omega = results['N'] * np.pi / 30
    phi_rad = np.radians(results['phi'])
    R = results['R']
    zeta_rad = np.radians(gamma)
    r_temp = R - (2 * R * np.tan(beta_rad) * np.tan(zeta_rad) * (1 - np.cos(phi_rad)))
    FwK = results['mass'] * omega**2 * r_temp

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    plt.plot(results['phi'], FwK, label=r'FwK - Centrifugal Force (N)')
    plt.plot(results['phi'], results['Inertial_Force'][gamma], label=r'FaK - Inertial Force (N)')

    plt.xlabel(r'Shaft Angle ($^\circ$)')
    plt.ylabel(r'Force (N)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 360) # Explicitly set x-axis limits
    plt.tight_layout()
    plt.savefig('centrifugal_inertial_forces.png')


def plot_radial_force_FSKy(results, gamma):
    """
    Plots FSKy (Radial Force) for a given gamma on a separate plot,
    formatted for LaTeX, without a title.
    """
    if gamma not in results['gamma']:
        print(f"Error: Gamma value {gamma} not found in calculated results.")
        return

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    plt.plot(results['phi'], results['FSKy'][gamma], label=r'FSKy - Radial Force (N)', color='purple')

    plt.xlabel(r'Shaft Angle ($^\circ$)')
    plt.ylabel(r'Force (N)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 360) # Explicitly set x-axis limits

    plt.tight_layout()
    plt.savefig('radial_force_FSKy.png')



def plot_motion_quantities(results, gamma):
    """
    Plots Acceleration, Velocity (integrated), and Displacement (integrated) vs Shaft Angle (phi)
    for a given gamma value. No gamma label in plot.
    """
    if gamma not in results['gamma']:
        print(f"Error: Gamma value {gamma} not found in calculated results.")
        return

    phi = results['phi']
    acc = results['Acceleration'][gamma]

    # Use simple cumulative sum to simulate velocity and displacement
    # (not physical, just to visualize trends over shaft angle)
    velocity = np.cumsum(acc) * (phi[1] - phi[0])
    displacement = np.cumsum(velocity) * (phi[1] - phi[0])

    plt.figure(figsize=(10, 6))
    plt.plot(phi, acc, label='Acceleration (m/s²)')
    plt.plot(phi, velocity, label='Velocity (arb. units)')
    plt.plot(phi, displacement, label='Displacement (arb. units)')

    plt.xlabel("Shaft Angle (°)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 360)
    plt.tight_layout()
    plt.savefig('motion_quantities.png')



# === User Input Section ===
# Ensure 'P_DC_subset.csv' is in the same directory as this script,
# or provide the full path to your file.
pdc_file_path = "../Results/P_DC_subset.csv"

# Static input parameters
R = 42.08 * 0.001       # meters (Radius of the main shaft, corresponds to rB in C++ snippet)
r = 9.73 * 0.001        # meters (Outer radius of the piston)
r_inner = 1.5 * 0.001   # meters (Inner radius of the piston)
Alpha = 14              # degrees (Swashplate angle, corresponds to beta in C++ snippet)
gamma_values = [0] # degrees (Swashplate cross angle, corresponds to gamma in C++ snippet)
N = 4500                # RPM (Rotational speed)
mass = 0.103117         # kg (Mass of the piston/slipper assembly, corresponds to mK in C++ snippet)

# === Load PDC Data ===
try:
    pdc_df = pd.read_csv(pdc_file_path)
    if 'PDC' not in pdc_df.columns:
        raise ValueError("CSV file must have a column named 'PDC'.")
    pdc_data = pdc_df['PDC'].to_numpy()
except FileNotFoundError:
    print(f"Error: The file '{pdc_file_path}' was not found.")
    print("Please make sure the CSV file is in the correct directory or provide the full path.")
    exit(1)
except Exception as e:
    print(f"Error loading PDC data: {e}")
    exit(1)

# Generate shaft angle phi based on the length of PDC data
# Use endpoint=False to ensure the range goes from 0 up to, but not including, 360.
# This is common if the data for 360 degrees is effectively the same as 0 degrees.
phi = np.linspace(0, 360, len(pdc_data), endpoint=False)


# === Execute Everything ===
print("Starting piston dynamics calculation...")
results = calculate_piston_dynamics(R, r, r_inner, Alpha, gamma_values, N, mass, pdc_data, phi)
print("Calculation complete. Exporting results to ZIP...")
export_results_to_zip(results)

# === Plot Results ===
print("\nGenerating plots...")
# Plot individual force series for all gamma values (optional, as per original code)
# plot_series(results, 'FAKz', 'FAKz (N)')
# plot_series(results, 'QuerKraft', 'QuerKraft (N)')


# plot_motion_quantities(results, gamma=0)
# # Plot all forces for gamma = 0 on a single plot (as requested)
plot_all_forces_at_gamma_zero(results)
# print("Plots generated.")
#
# plot_pressure_force(results)
# plot_total_piston_force(results, gamma=0)
# plot_reaction_force(results, gamma=0)
# plot_centrifugal_force(results, gamma=0)
# plot_inertial_force(results, gamma=0)
#
# # Generate the requested separate plots
# print("\nGenerating separate plots for Centrifugal/Inertial and Radial Forces...")
# plot_centrifugal_and_inertial_forces(results, gamma=0)
# plot_radial_force_FSKy(results, gamma=0)
# print("Separate plots generated.")