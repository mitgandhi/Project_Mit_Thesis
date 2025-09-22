# import numpy as np
# import pandas as pd
# import math
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.gridspec import GridSpec
#
#
# class GeometryPistonGap:
#     def __init__(self):
#         self.mK = 0.103  # Mass piston/slipper assembly (kg)
#         self.lSK = 20.0  # Distance piston center of mass/slipper assembly from piston head (mm)
#         self.zRK = 10.0  # Distance between piston head and beginning of the gap (mm)
#
#
# class OperatingPistonGap:
#     def __init__(self):
#         self.beta_rad = np.radians(14)  # Swashplate angle (15 degrees in radians)
#         self.betamax_rad = np.radians(14)  # Max swashplate angle (radians)
#         self.gamma_rad = 0.0  # Swashplate cross angle (radians)
#         self.omega = np.pi * 4400 / 30  # Angular speed (rad/s) - 1500 RPM
#         self.pCase = 100000  # Case pressure (Pa) - atmospheric
#
#
# class ForcesPistonGap:
#     def __init__(self):
#         self.FTG = 0.0  # Slipper friction force
#         self.FTKy = [0.0]  # Friction force axial direction array
#         # Results will be stored here
#         self.FaK = 0.0
#         self.FDK = 0.0
#         self.Fwk = 0.0
#         self.FKx = 0.0
#         self.FKy = 0.0
#         self.FK = 0.0
#         self.MKx = 0.0
#         self.MKy = 0.0
#         self.MK = 0.0
#         self.FSKx = 0.0
#         self.FSKy = 0.0
#         self.FAKz = 0.0
#         self.FAKy = 0.0
#         self.Fwky = 0.0
#         self.Fwkz = 0.0
#         self.FAK_inclined = 0.0
#
#
# class MyInput:
#     def __init__(self):
#         self.friction_less = False  # Set to True to simulate friction-less conditions
#         self.zeta_deg = 0.0  # Piston inclination angle in degrees
#
#
# def piston_calc_external_forces_inclined_piston(phi_deg, pDC, geometry, operating, forces, my_input,
#                                                 AreaK, rB):
#     """
#     Calculate external forces on an inclined piston for a given angular position and pressure.
#
#     Parameters:
#     phi_deg: Angular position in degrees (0-360)
#     pDC: Displacement chamber pressure (Pa)
#     geometry: GeometryPistonGap object
#     operating: OperatingPistonGap object
#     forces: ForcesPistonGap object
#     my_input: MyInput object
#     AreaK: Piston area (mm²)
#     rB: Pitch circle radius (mm)
#
#     Returns:
#     Updated forces object with calculated forces and moments
#     """
#
#     # Convert degrees to radians
#     phi = math.radians(phi_deg)
#     PI = math.pi
#
#     # Variable initialization from objects
#     mK = geometry.mK  # kg
#     lSK = geometry.lSK / 1000.0  # Convert mm to m for calculations
#     zRK = geometry.zRK / 1000.0  # Convert mm to m for calculations
#     beta = operating.beta_rad
#     betamax = operating.betamax_rad
#     gamma = operating.gamma_rad
#     omega = operating.omega
#     pCase = operating.pCase
#     FTG = forces.FTG
#     zeta = math.radians(my_input.zeta_deg)
#
#     # Convert mm to m for calculations
#     AreaK_m2 = AreaK / 1000000.0  # mm² to m²
#     rB_m = rB / 1000.0  # mm to m
#
#     # Pressure force
#     FDK = AreaK_m2 * (pDC - pCase)
#
#     # Calculate r_temp
#     r_temp = rB_m - (2 * rB_m * math.tan(beta) * math.tan(zeta) *(1-math.cos(phi)))
#
#     # Center of mass inertia force z direction
#     FaKz = mK * (omega * omega) * r_temp * (math.tan(beta) / math.cos(zeta)) * math.cos(phi)
#
#     # Centrifugal force (set to zero to simulate EHD test rig)
#     FwK_inclined = mK * (omega * omega) * r_temp
#
#     # Centrifugal components for transforming inclined piston frame to global frame
#     Fwkz = FwK_inclined * math.sin(zeta)
#     Fwky = FwK_inclined * math.cos(zeta)
#
#     # Store basic forces
#     forces.FaK = FaKz
#     forces.FDK = FDK
#     forces.Fwk = FwK_inclined
#
#     # Handle friction forces
#     if my_input.friction_less:
#         # Friction force axial direction
#         FTKy = 0.0
#         # Slipper friction force tangential direction
#         FTGx = 0.0
#         # Slipper friction force y direction
#         FTGy = 0.0
#     else:
#         # Friction force axial direction
#         FTKy = sum(forces.FTKy)
#         # Slipper friction force tangential direction
#         FTGx = -FTG
#         # Slipper friction force y direction
#         FTGy = 0.0
#
#     # Forces on the piston acting in inclined plane
#     FAK_inclined = FDK + FaKz + FTKy
#
#     # From inclined plane frame to global frame
#     FAKz = FAK_inclined * math.cos(zeta)
#     FAKy = FAK_inclined * math.sin(zeta)
#
#     # Reaction forces for inclined
#     FSK = FAKz / math.cos(beta)
#
#     # Radial Component of FSK
#     FSKy = FSK * math.sin(beta)
#     FSKx = 0.0  # This variable was used but not defined in original code
#
#     # External force components (x and y directions)
#     forces.FKx = -(FSKy + FAKy) * math.sin(phi) + FSKx * math.cos(phi) + FTGx
#     forces.FKy = (FSKy + FAKy) * math.cos(phi) + FSKx * math.sin(phi) + Fwky + FTGy
#
#     FKx = forces.FKx
#     FKy = forces.FKy
#
#     # Total external force
#     forces.FK = math.sqrt((FKx * FKx) + (FKy * FKy))
#
#     # External moment components (x and y directions) - results in N⋅mm
#     forces.MKx = -zRK * ((FSKy + FAKy) * math.cos(phi) + FSKx * math.sin(phi)) - (zRK - lSK) * Fwky
#     forces.MKy = zRK * FKx
#
#     # Convert moments to N⋅mm for output
#     forces.MKx = forces.MKx  # N⋅m
#     forces.MKy = forces.MKy  # N⋅m
#
#     MKx = forces.MKx
#     MKy = forces.MKy
#
#     # Total external moment
#     forces.MK = math.sqrt((MKx * MKx) + (MKy * MKy))
#
#     # Store additional results
#     forces.FSKx = FSKx
#     forces.FSKy = FSKy
#     forces.FAKz = FAKz
#     forces.FAKy = FAKy
#     forces.Fwky = Fwky
#     forces.Fwkz = Fwkz
#     forces.FAK_inclined = FAK_inclined
#
#     return forces
#
#
# def calculate_forces_for_full_rotation(pdc_file_path, output_file_path=None):
#     """
#     Calculate forces for a full rotation (0-360 degrees) using pDC data from CSV file.
#
#     Parameters:
#     pdc_file_path: Path to CSV file containing pDC values
#     output_file_path: Optional path to save results CSV
#
#     Returns:
#     DataFrame with results for all angular positions
#     """
#
#     # Read pDC data from CSV
#     pdc_df = pd.read_csv(pdc_file_path)
#     pdc_values = pdc_df['PDC'].values
#
#     # Ensure we have 360 values
#     if len(pdc_values) != 360:
#         raise ValueError(f"Expected 360 pDC values, got {len(pdc_values)}")
#
#     # Initialize objects
#     geometry = GeometryPistonGap()
#     operating = OperatingPistonGap()
#     forces = ForcesPistonGap()
#     my_input = MyInput()
#
#     # Parameters
#     r_piston = 9.75
#     rB = 42.08  # mm - pitch circle radius
#     AreaK = (np.pi * r_piston * r_piston) - (np.pi*1.5*1.5) # mm² - piston area
#
#     # Storage for results
#     results = []
#
#     # Calculate for each degree
#     for phi_deg in range(360):
#         pDC = pdc_values[phi_deg]
#
#         # Calculate forces for this position
#         forces_result = piston_calc_external_forces_inclined_piston(
#             phi_deg, pDC, geometry, operating, forces, my_input, AreaK, rB
#         )
#
#         # Store results
#         result = {
#             'phi_deg': phi_deg,
#             'pDC': pDC,
#             'FDK': forces_result.FDK,
#             'FaK': forces_result.FaK,
#             'Fwk': forces_result.Fwk,
#             'FKx': forces_result.FKx,
#             'FKy': forces_result.FKy,
#             'FK': forces_result.FK,
#             'MKx': forces_result.MKx,
#             'MKy': forces_result.MKy,
#             'MK': forces_result.MK,
#             'FSKx': forces_result.FSKx,
#             'FSKy': forces_result.FSKy,
#             'FAKz': forces_result.FAKz,
#             'FAKy': forces_result.FAKy,
#             'Fwky': forces_result.Fwky,
#             'Fwkz': forces_result.Fwkz,
#             'FAK_inclined': forces_result.FAK_inclined
#         }
#         results.append(result)
#
#     # Convert to DataFrame
#     results_df = pd.DataFrame(results)
#
#     # Save to file if requested
#     if output_file_path:
#         results_df.to_csv(output_file_path, index=False)
#         print(f"Results saved to {output_file_path}")
#
#     return results_df
#
#
# def calculate_forces_for_multiple_zeta(pdc_file_path, zeta_angles=[0, 5, 10, 15], output_file_path=None):
#     """
#     Calculate forces for multiple zeta angles across a full rotation.
#
#     Parameters:
#     pdc_file_path: Path to CSV file containing pDC values
#     zeta_angles: List of zeta angles in degrees (default: [0, 5, 10, 15])
#     output_file_path: Optional path to save results CSV
#
#     Returns:
#     DataFrame with results for all angular positions and zeta angles
#     """
#
#     # Read pDC data from CSV
#     pdc_df = pd.read_csv(pdc_file_path)
#     pdc_values = pdc_df['PDC'].values
#
#     # Ensure we have 360 values
#     if len(pdc_values) != 360:
#         raise ValueError(f"Expected 360 pDC values, got {len(pdc_values)}")
#
#     # Parameters
#     rB = 42.08  # mm - pitch circle radius
#     r_piston = 9.75  # mm
#     AreaK = np.pi * r_piston * r_piston  # mm² - piston area
#
#     # Storage for all results
#     all_results = []
#
#     # Calculate for each zeta angle
#     for zeta_deg in zeta_angles:
#         print(f"Calculating forces for zeta = {zeta_deg}°...")
#
#         # Initialize objects for this zeta
#         geometry = GeometryPistonGap()
#         operating = OperatingPistonGap()
#         forces = ForcesPistonGap()
#         my_input = MyInput()
#         my_input.zeta_deg = zeta_deg  # Set the zeta angle
#
#         # Calculate for each phi degree
#         for phi_deg in range(360):
#             pDC = pdc_values[phi_deg]
#
#             # Calculate forces for this position and zeta
#             forces_result = piston_calc_external_forces_inclined_piston(
#                 phi_deg, pDC, geometry, operating, forces, my_input, AreaK, rB
#             )
#
#             # Store results with zeta information
#             result = {
#                 'zeta_deg': zeta_deg,
#                 'phi_deg': phi_deg,
#                 'pDC': pDC,
#                 'FDK': forces_result.FDK,
#                 'FaK': forces_result.FaK,
#                 'Fwk': forces_result.Fwk,
#                 'FKx': forces_result.FKx,
#                 'FKy': forces_result.FKy,
#                 'FK': forces_result.FK,
#                 'MKx': forces_result.MKx,
#                 'MKy': forces_result.MKy,
#                 'MK': forces_result.MK,
#                 'FSKx': forces_result.FSKx,
#                 'FSKy': forces_result.FSKy,
#                 'FAKz': forces_result.FAKz,
#                 'FAKy': forces_result.FAKy,
#                 'Fwky': forces_result.Fwky,
#                 'Fwkz': forces_result.Fwkz,
#                 'FAK_inclined': forces_result.FAK_inclined
#             }
#             all_results.append(result)
#
#     # Convert to DataFrame
#     results_df = pd.DataFrame(all_results)
#
#     # Save to file if requested
#     if output_file_path:
#         results_df.to_csv(output_file_path, index=False)
#         print(f"Results saved to {output_file_path}")
#
#     # Print summary for each zeta
#     print("\nSummary Statistics by Zeta Angle:")
#     print("=" * 60)
#     for zeta_deg in zeta_angles:
#         zeta_data = results_df[results_df['zeta_deg'] == zeta_deg]
#         print(f"\nZeta = {zeta_deg}°:")
#         print(f"  Max total force FK: {zeta_data['FK'].max():.2f} N")
#         print(f"  Min total force FK: {zeta_data['FK'].min():.2f} N")
#         print(f"  Mean total force FK: {zeta_data['FK'].mean():.2f} N")
#         print(f"  Max total moment MK: {zeta_data['MK'].max():.2f} N⋅mm")
#         print(f"  Min total moment MK: {zeta_data['MK'].min():.2f} N⋅mm")
#         print(f"  Mean total moment MK: {zeta_data['MK'].mean():.2f} N⋅mm")
#
#     return results_df
#
#
# def plot_forces_vs_phi(results_df, save_plots=True, plot_dir='plots'):
#     """
#     Create comprehensive plots showing forces vs phi angle for different zeta values.
#
#     Parameters:
#     results_df: DataFrame from calculate_forces_for_multiple_zeta
#     save_plots: Boolean, whether to save plots to files
#     plot_dir: Directory to save plots
#     """
#
#     # Set up the plotting style
#     plt.style.use('seaborn-v0_8')
#     sns.set_palette("husl")
#
#     # Create directory for plots if saving
#     if save_plots:
#         import os
#         os.makedirs(plot_dir, exist_ok=True)
#
#     # Get unique zeta angles
#     zeta_angles = sorted(results_df['zeta_deg'].unique())
#
#     # Define color palette
#     colors = plt.cm.tab10(np.linspace(0, 1, len(zeta_angles)))
#     color_dict = {zeta: colors[i] for i, zeta in enumerate(zeta_angles)}
#
#     # 1. COMPREHENSIVE FORCES PLOT (2x3 grid)
#     fig1 = plt.figure(figsize=(18, 12))
#     gs1 = GridSpec(2, 3, figure=fig1, hspace=0.3, wspace=0.3)
#
#     force_plots = [
#         ('FK', 'Total Force FK [N]', (0, 0)),
#         ('FKx', 'Force FKx [N]', (0, 1)),
#         ('FKy', 'Force FKy [N]', (0, 2)),
#         ('FDK', 'Pressure Force FDK [N]', (1, 0)),
#         ('FaK', 'Inertia Force FaK [N]', (1, 1)),
#         ('Fwk', 'Centrifugal Force Fwk [N]', (1, 2))
#     ]
#
#     for force_name, ylabel, (row, col) in force_plots:
#         ax = fig1.add_subplot(gs1[row, col])
#
#         for zeta in zeta_angles:
#             zeta_data = results_df[results_df['zeta_deg'] == zeta]
#             ax.plot(zeta_data['phi_deg'], zeta_data[force_name],
#                     label=f'ζ = {zeta}°', linewidth=2, color=color_dict[zeta])
#
#         ax.set_xlabel('Phi Angle [°]')
#         ax.set_ylabel(ylabel)
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#         ax.set_xlim(0, 360)
#
#         # Set x-ticks at 60-degree intervals
#         ax.set_xticks(np.arange(0, 361, 60))
#
#     fig1.suptitle('Piston Forces vs Phi Angle for Different Zeta Values', fontsize=16, fontweight='bold')
#
#     if save_plots:
#         fig1.savefig(f'{plot_dir}/forces_vs_phi_comprehensive.png', dpi=300, bbox_inches='tight')
#         fig1.savefig(f'{plot_dir}/forces_vs_phi_comprehensive.pdf', bbox_inches='tight')
#
#     # 2. MOMENTS PLOT
#     fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
#
#     moment_plots = [
#         ('MK', 'Total Moment MK [N⋅m]', 0),
#         ('MKx', 'Moment MKx [N⋅m]', 1),
#         ('MKy', 'Moment MKy [N⋅m]', 2)
#     ]
#
#     for moment_name, ylabel, idx in moment_plots:
#         ax = axes2[idx]
#
#         for zeta in zeta_angles:
#             zeta_data = results_df[results_df['zeta_deg'] == zeta]
#             ax.plot(zeta_data['phi_deg'], zeta_data[moment_name],
#                     label=f'ζ = {zeta}°', linewidth=2, color=color_dict[zeta])
#
#         ax.set_xlabel('Phi Angle [°]')
#         ax.set_ylabel(ylabel)
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#         ax.set_xlim(0, 360)
#         ax.set_xticks(np.arange(0, 361, 60))
#
#     fig2.suptitle('Piston Moments vs Phi Angle for Different Zeta Values', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#
#     if save_plots:
#         fig2.savefig(f'{plot_dir}/moments_vs_phi.png', dpi=300, bbox_inches='tight')
#         fig2.savefig(f'{plot_dir}/moments_vs_phi.pdf', bbox_inches='tight')
#
#     # 3. ADDITIONAL FORCES PLOT
#     fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
#     axes3 = axes3.flatten()
#
#     additional_forces = [
#         ('FSKy', 'Slipper Force FSKy [N]'),
#         ('FAKz', 'Axial Force FAKz [N]'),
#         ('FAKy', 'Radial Force FAKy [N]'),
#         ('Fwky', 'Centrifugal Force Fwky [N]'),
#         ('Fwkz', 'Centrifugal Force Fwkz [N]'),
#         ('FAK_inclined', 'Inclined Force FAK [N]')
#     ]
#
#     for idx, (force_name, ylabel) in enumerate(additional_forces):
#         ax = axes3[idx]
#
#         for zeta in zeta_angles:
#             zeta_data = results_df[results_df['zeta_deg'] == zeta]
#             ax.plot(zeta_data['phi_deg'], zeta_data[force_name],
#                     label=f'ζ = {zeta}°', linewidth=2, color=color_dict[zeta])
#
#         ax.set_xlabel('Phi Angle [°]')
#         ax.set_ylabel(ylabel)
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#         ax.set_xlim(0, 360)
#         ax.set_xticks(np.arange(0, 361, 60))
#
#     fig3.suptitle('Additional Piston Forces vs Phi Angle for Different Zeta Values', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#
#     if save_plots:
#         fig3.savefig(f'{plot_dir}/additional_forces_vs_phi.png', dpi=300, bbox_inches='tight')
#         fig3.savefig(f'{plot_dir}/additional_forces_vs_phi.pdf', bbox_inches='tight')
#
#     # 4. POLAR PLOT FOR TOTAL FORCE
#     fig4, ax4 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
#
#     for zeta in zeta_angles:
#         zeta_data = results_df[results_df['zeta_deg'] == zeta]
#         phi_rad = np.radians(zeta_data['phi_deg'])
#         ax4.plot(phi_rad, zeta_data['FK'], label=f'ζ = {zeta}°',
#                  linewidth=2, color=color_dict[zeta])
#
#     ax4.set_title('Total Force FK - Polar Plot\n(Radial: Force [N], Angular: Phi [°])',
#                   fontsize=14, fontweight='bold', pad=20)
#     ax4.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))
#     ax4.grid(True)
#
#     if save_plots:
#         fig4.savefig(f'{plot_dir}/forces_polar_plot.png', dpi=300, bbox_inches='tight')
#         fig4.savefig(f'{plot_dir}/forces_polar_plot.pdf', bbox_inches='tight')
#
#     # 5. COMPARISON PLOT - NORMALIZED FORCES
#     fig5, ax5 = plt.subplots(figsize=(14, 8))
#
#     for zeta in zeta_angles:
#         zeta_data = results_df[results_df['zeta_deg'] == zeta]
#         # Normalize by maximum value for comparison
#         fk_normalized = zeta_data['FK'] / zeta_data['FK'].max()
#         ax5.plot(zeta_data['phi_deg'], fk_normalized,
#                  label=f'ζ = {zeta}°', linewidth=2, color=color_dict[zeta])
#
#     ax5.set_xlabel('Phi Angle [°]', fontsize=12)
#     ax5.set_ylabel('Normalized Total Force FK [-]', fontsize=12)
#     ax5.set_title('Normalized Total Force Comparison\n(Each curve normalized by its maximum)',
#                   fontsize=14, fontweight='bold')
#     ax5.grid(True, alpha=0.3)
#     ax5.legend()
#     ax5.set_xlim(0, 360)
#     ax5.set_xticks(np.arange(0, 361, 60))
#
#     if save_plots:
#         fig5.savefig(f'{plot_dir}/normalized_forces_comparison.png', dpi=300, bbox_inches='tight')
#         fig5.savefig(f'{plot_dir}/normalized_forces_comparison.pdf', bbox_inches='tight')
#
#     plt.show()
#
#     if save_plots:
#         print(f"\nAll plots saved to '{plot_dir}/' directory")
#         print("Generated files:")
#         print("- forces_vs_phi_comprehensive.png/pdf")
#         print("- moments_vs_phi.png/pdf")
#         print("- additional_forces_vs_phi.png/pdf")
#         print("- forces_polar_plot.png/pdf")
#         print("- normalized_forces_comparison.png/pdf")
#
#
# def plot_individual_force_comparisons(results_df, save_plots=True, plot_dir='plots'):
#     """
#     Create individual plots for each force type with all zeta angles.
#
#     Parameters:
#     results_df: DataFrame from calculate_forces_for_multiple_zeta
#     save_plots: Boolean, whether to save plots to files
#     plot_dir: Directory to save plots
#     """
#
#     # Set up the plotting style
#     plt.style.use('seaborn-v0_8')
#
#     # Create directory for plots if saving
#     if save_plots:
#         import os
#         os.makedirs(f'{plot_dir}/individual_forces', exist_ok=True)
#
#     # Get unique zeta angles
#     zeta_angles = sorted(results_df['zeta_deg'].unique())
#
#     # Define color palette
#     colors = plt.cm.tab10(np.linspace(0, 1, len(zeta_angles)))
#     color_dict = {zeta: colors[i] for i, zeta in enumerate(zeta_angles)}
#
#     # List of all forces to plot individually
#     forces_to_plot = [
#         ('FK', 'Total Force FK [N]'),
#         ('FKx', 'Force FKx [N]'),
#         ('FKy', 'Force FKy [N]'),
#         ('FDK', 'Pressure Force FDK [N]'),
#         ('FaK', 'Inertia Force FaK [N]'),
#         ('Fwk', 'Centrifugal Force Fwk [N]'),
#         ('MK', 'Total Moment MK [N⋅m]'),
#         ('MKx', 'Moment MKx [N⋅m]'),
#         ('MKy', 'Moment MKy [N⋅m]'),
#         ('FSKy', 'Slipper Force FSKy [N]'),
#         ('FAKz', 'Axial Force FAKz [N]'),
#         ('FAKy', 'Radial Force FAKy [N]'),
#         ('Fwky', 'Centrifugal Force Fwky [N]'),
#         ('Fwkz', 'Centrifugal Force Fwkz [N]'),
#         ('FAK_inclined', 'Inclined Force FAK [N]')
#     ]
#
#     for force_name, ylabel in forces_to_plot:
#         fig, ax = plt.subplots(figsize=(12, 8))
#
#         # Plot for each zeta angle
#         for zeta in zeta_angles:
#             zeta_data = results_df[results_df['zeta_deg'] == zeta]
#             ax.plot(zeta_data['phi_deg'], zeta_data[force_name],
#                     label=f'ζ = {zeta}°', linewidth=2.5, color=color_dict[zeta])
#
#         ax.set_xlabel('Phi Angle [°]', fontsize=12)
#         ax.set_ylabel(ylabel, fontsize=12)
#         ax.set_title(f'{ylabel} vs Phi Angle for Different Zeta Values',
#                      fontsize=14, fontweight='bold')
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=11)
#         ax.set_xlim(0, 360)
#         ax.set_xticks(np.arange(0, 361, 60))
#
#         # Add some statistics as text
#         max_vals = []
#         min_vals = []
#         for zeta in zeta_angles:
#             zeta_data = results_df[results_df['zeta_deg'] == zeta]
#             max_vals.append(zeta_data[force_name].max())
#             min_vals.append(zeta_data[force_name].min())
#
#         stats_text = f'Max range: {min(min_vals):.2f} to {max(max_vals):.2f}'
#         ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
#                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
#
#         plt.tight_layout()
#
#         if save_plots:
#             filename_base = force_name.lower().replace('_', '')
#             fig.savefig(f'{plot_dir}/individual_forces/{filename_base}_vs_phi.png',
#                         dpi=300, bbox_inches='tight')
#             fig.savefig(f'{plot_dir}/individual_forces/{filename_base}_vs_phi.pdf',
#                         bbox_inches='tight')
#
#         plt.show()
#
#     if save_plots:
#         print(f"\nIndividual force plots saved to '{plot_dir}/individual_forces/' directory")
#
#
# def analyze_zeta_effects(results_df):
#     """
#     Analyze the effects of different zeta angles on piston forces.
#
#     Parameters:
#     results_df: DataFrame from calculate_forces_for_multiple_zeta
#
#     Returns:
#     DataFrame with comparative analysis
#     """
#
#     # Group by zeta angle and calculate statistics
#     zeta_summary = results_df.groupby('zeta_deg').agg({
#         'FK': ['min', 'max', 'mean', 'std'],
#         'MK': ['min', 'max', 'mean', 'std'],
#         'FKx': ['min', 'max', 'mean', 'std'],
#         'FKy': ['min', 'max', 'mean', 'std'],
#         'FAKz': ['min', 'max', 'mean', 'std'],
#         'FAKy': ['min', 'max', 'mean', 'std']
#     }).round(6)
#
#     # Flatten column names
#     zeta_summary.columns = ['_'.join(col).strip() for col in zeta_summary.columns.values]
#     zeta_summary = zeta_summary.reset_index()
#
#     print("\nDetailed Zeta Angle Effects Analysis:")
#     print("=" * 80)
#     print(zeta_summary.to_string(index=False))
#
#     return zeta_summary
#
#
# def create_summary_plots(results_df, save_plots=True, plot_dir='plots'):
#     """
#     Create summary plots showing key statistics and comparisons.
#
#     Parameters:
#     results_df: DataFrame from calculate_forces_for_multiple_zeta
#     save_plots: Boolean, whether to save plots to files
#     plot_dir: Directory to save plots
#     """
#
#     # Set up the plotting style
#     plt.style.use('seaborn-v0_8')
#
#     # Create directory for plots if saving
#     if save_plots:
#         import os
#         os.makedirs(f'{plot_dir}/summary', exist_ok=True)
#
#     # Get unique zeta angles
#     zeta_angles = sorted(results_df['zeta_deg'].unique())
#
#     # 1. SUMMARY STATISTICS BAR PLOT
#     fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
#     axes1 = axes1.flatten()
#
#     force_stats = ['FK', 'FKx', 'FKy', 'MK', 'MKx', 'MKy']
#     stat_types = ['max', 'min', 'mean']
#
#     for idx, force in enumerate(force_stats):
#         ax = axes1[idx]
#
#         # Calculate statistics for each zeta
#         max_vals = []
#         min_vals = []
#         mean_vals = []
#
#         for zeta in zeta_angles:
#             zeta_data = results_df[results_df['zeta_deg'] == zeta]
#             max_vals.append(zeta_data[force].max())
#             min_vals.append(zeta_data[force].min())
#             mean_vals.append(zeta_data[force].mean())
#
#         x = np.arange(len(zeta_angles))
#         width = 0.25
#
#         ax.bar(x - width, max_vals, width, label='Max', alpha=0.8)
#         ax.bar(x, mean_vals, width, label='Mean', alpha=0.8)
#         ax.bar(x + width, min_vals, width, label='Min', alpha=0.8)
#
#         ax.set_xlabel('Zeta Angle [°]')
#         ax.set_ylabel(f'{force} Statistics')
#         ax.set_title(f'{force} Statistics by Zeta Angle')
#         ax.set_xticks(x)
#         ax.set_xticklabels([f'{z}°' for z in zeta_angles])
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#
#     fig1.suptitle('Force and Moment Statistics by Zeta Angle', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#
#     if save_plots:
#         fig1.savefig(f'{plot_dir}/summary/statistics_by_zeta.png', dpi=300, bbox_inches='tight')
#         fig1.savefig(f'{plot_dir}/summary/statistics_by_zeta.pdf', bbox_inches='tight')
#
#     # 2. HEATMAP OF FORCE VALUES
#     fig2, ax2 = plt.subplots(figsize=(14, 10))
#
#     # Create pivot table for heatmap
#     pivot_data = results_df.pivot_table(values='FK', index='phi_deg', columns='zeta_deg')
#
#     im = ax2.imshow(pivot_data.T, aspect='auto', cmap='viridis', origin='lower')
#
#     # Set ticks and labels
#     ax2.set_xticks(np.arange(0, 360, 30))
#     ax2.set_xticklabels(np.arange(0, 360, 30))
#     ax2.set_yticks(np.arange(len(zeta_angles)))
#     ax2.set_yticklabels([f'{z}°' for z in zeta_angles])
#
#     ax2.set_xlabel('Phi Angle [°]')
#     ax2.set_ylabel('Zeta Angle')
#     ax2.set_title('Total Force FK Heatmap\n(Color represents force magnitude)')
#
#     # Add colorbar
#     cbar = plt.colorbar(im, ax=ax2)
#     cbar.set_label('Total Force FK [N]')
#
#     if save_plots:
#         fig2.savefig(f'{plot_dir}/summary/force_heatmap.png', dpi=300, bbox_inches='tight')
#         fig2.savefig(f'{plot_dir}/summary/force_heatmap.pdf', bbox_inches='tight')
#
#     # 3. PHASE ANALYSIS PLOT
#     fig3, ax3 = plt.subplots(figsize=(12, 8))
#
#     # Find phase shift for each zeta (where FK is maximum)
#     colors = plt.cm.tab10(np.linspace(0, 1, len(zeta_angles)))
#
#     for i, zeta in enumerate(zeta_angles):
#         zeta_data = results_df[results_df['zeta_deg'] == zeta]
#         max_idx = zeta_data['FK'].idxmax()
#         max_phi = zeta_data.loc[max_idx, 'phi_deg']
#         max_force = zeta_data.loc[max_idx, 'FK']
#
#         ax3.scatter(max_phi, max_force, s=200, color=colors[i],
#                     label=f'ζ = {zeta}° (φ = {max_phi}°)', alpha=0.8, edgecolors='black')
#
#     ax3.set_xlabel('Phi Angle at Maximum Force [°]')
#     ax3.set_ylabel('Maximum Force [N]')
#     ax3.set_title('Maximum Force Location and Magnitude by Zeta Angle')
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)
#
#     if save_plots:
#         fig3.savefig(f'{plot_dir}/summary/max_force_analysis.png', dpi=300, bbox_inches='tight')
#         fig3.savefig(f'{plot_dir}/summary/max_force_analysis.pdf', bbox_inches='tight')
#
#     plt.show()
#
#     if save_plots:
#         print(f"\nSummary plots saved to '{plot_dir}/summary/' directory")
#
#
# # Example usage with plotting:
# if __name__ == "__main__":
#     # Method 1: Calculate forces for single zeta angle (original functionality)
#     print("=" * 60)
#     print("SINGLE ZETA CALCULATION (zeta = 0°)")
#     print("=" * 60)
#     results_single = calculate_forces_for_full_rotation('P_DC_subset.csv', 'piston_forces_single_zeta.csv')
#
#     # Display summary statistics for single zeta
#     print("Force Calculation Results Summary (zeta = 0°):")
#     print(f"Max total force FK: {results_single['FK'].max():.2f} N")
#     print(f"Min total force FK: {results_single['FK'].min():.2f} N")
#     print(f"Max total moment MK: {results_single['MK'].max():.6f} N⋅m")
#     print(f"Min total moment MK: {results_single['MK'].min():.6f} N⋅m")
#
#     print("\n" + "=" * 60)
#     print("MULTIPLE ZETA CALCULATIONS (0°, 5°, 10°, 15°)")
#     print("=" * 60)
#
#     # Method 2: Calculate forces for multiple zeta angles
#     zeta_angles = [0, 5, 10, 15]  # degrees
#     results_multi = calculate_forces_for_multiple_zeta(
#         'P_DC_subset.csv',
#         zeta_angles,
#         'piston_forces_multiple_zeta.csv'
#     )
#
#     # Analyze the effects of different zeta angles
#     zeta_analysis = analyze_zeta_effects(results_multi)
#
#     # Additional analysis: Find phi angles with maximum forces for each zeta
#     print("\n" + "=" * 60)
#     print("PHI ANGLES WITH MAXIMUM FORCES FOR EACH ZETA")
#     print("=" * 60)
#
#     for zeta_deg in zeta_angles:
#         zeta_data = results_multi[results_multi['zeta_deg'] == zeta_deg]
#         max_force_idx = zeta_data['FK'].idxmax()
#         max_moment_idx = zeta_data['MK'].idxmax()
#
#         max_force_phi = zeta_data.loc[max_force_idx, 'phi_deg']
#         max_force_value = zeta_data.loc[max_force_idx, 'FK']
#         max_moment_phi = zeta_data.loc[max_moment_idx, 'phi_deg']
#         max_moment_value = zeta_data.loc[max_moment_idx, 'MK']
#
#         print(f"\nZeta = {zeta_deg}°:")
#         print(f"  Max force FK = {max_force_value:.2f} N at phi = {max_force_phi}°")
#         print(f"  Max moment MK = {max_moment_value:.6f} N⋅m at phi = {max_moment_phi}°")
#
#     # Show sample data comparison
#     print("\n" + "=" * 60)
#     print("SAMPLE COMPARISON AT PHI = 90°")
#     print("=" * 60)
#
#     phi_90_data = results_multi[results_multi['phi_deg'] == 90][['zeta_deg', 'FK', 'MK', 'FKx', 'FKy']]
#     print(phi_90_data.to_string(index=False))
#
#     # CREATE ALL PLOTS
#     print("\n" + "=" * 60)
#     print("GENERATING PLOTS")
#     print("=" * 60)
#
#     # Main comprehensive plots
#     plot_forces_vs_phi(results_multi, save_plots=True, plot_dir='plots')
#
#     # Individual force plots (optional - uncomment if you want individual plots for each force)
#     # plot_individual_force_comparisons(results_multi, save_plots=True, plot_dir='plots')
#
#     # Summary and analysis plots
#     create_summary_plots(results_multi, save_plots=True, plot_dir='plots')
#
#     print("\n" + "=" * 60)
#     print("PLOTTING COMPLETE")
#     print("=" * 60)
#     print("Check the 'plots/' directory for all generated plots:")
#     print("- Comprehensive force plots")
#     print("- Moment plots")
#     print("- Additional force components")
#     print("- Polar plots")
#     print("- Summary statistics")
#     print("- Heatmaps and analysis plots")

import numpy as np
import pandas as pd
import math


class GeometryPistonGap:
    def __init__(self):
        self.mK = 0.103
        self.lSK = 20.0
        self.zRK = 10.0


class OperatingPistonGap:
    def __init__(self, speed_rpm=4400):
        self.beta_rad = np.radians(14)
        self.betamax_rad = np.radians(14)
        self.gamma_rad = 0.0
        self.omega = np.pi * speed_rpm / 30
        self.speed_rpm = speed_rpm
        self.pCase = 100000


class ForcesPistonGap:
    def __init__(self):
        self.FTG = 0.0
        self.FTKy = [0.0]
        self.FaK = 0.0
        self.FDK = 0.0
        self.Fwk = 0.0
        self.FKx = 0.0
        self.FKy = 0.0
        self.FK = 0.0
        self.MKx = 0.0
        self.MKy = 0.0
        self.MK = 0.0
        self.FSKx = 0.0
        self.FSKy = 0.0
        self.FAKz = 0.0
        self.FAKy = 0.0
        self.Fwky = 0.0
        self.Fwkz = 0.0
        self.FAK_inclined = 0.0


class MyInput:
    def __init__(self):
        self.friction_less = False
        self.zeta_deg = 0.0


def piston_calc_external_forces_inclined_piston(phi_deg, pDC, geometry, operating, forces, my_input, AreaK, rB):
    phi = math.radians(phi_deg)
    mK = geometry.mK
    lSK = geometry.lSK / 1000.0
    zRK = geometry.zRK / 1000.0
    beta = operating.beta_rad
    betamax = operating.betamax_rad
    gamma = operating.gamma_rad
    omega = operating.omega
    pCase = operating.pCase
    FTG = forces.FTG
    zeta = math.radians(my_input.zeta_deg)

    AreaK_m2 = AreaK / 1000000.0
    rB_m = rB / 1000.0

    FDK = AreaK_m2 * (pDC - pCase)
    r_temp = rB_m - (2 * rB_m * math.tan(beta) * math.tan(zeta) * (1 - math.cos(phi)))
    FaKz = mK * (omega * omega) * r_temp * (math.tan(beta) / math.cos(zeta)) * math.cos(phi)
    FwK_inclined = mK * (omega * omega) * r_temp
    Fwkz = FwK_inclined * math.sin(zeta)
    Fwky = FwK_inclined * math.cos(zeta)

    forces.FaK = FaKz
    forces.FDK = FDK
    forces.Fwk = FwK_inclined

    if my_input.friction_less:
        FTKy = 0.0
        FTGx = 0.0
        FTGy = 0.0
    else:
        FTKy = sum(forces.FTKy)
        FTGx = -FTG
        FTGy = 0.0

    FAK_inclined = FDK + FaKz + FTKy
    FAKz = FAK_inclined * math.cos(zeta)
    FAKy = FAK_inclined * math.sin(zeta)
    FSK = FAKz / math.cos(beta)
    FSKy = FSK * math.sin(beta)
    FSKx = 0.0

    forces.FKx = -(FSKy + FAKy) * math.sin(phi) + FSKx * math.cos(phi) + FTGx
    forces.FKy = (FSKy + FAKy) * math.cos(phi) + FSKx * math.sin(phi) + Fwky + FTGy

    FKx = forces.FKx
    FKy = forces.FKy
    forces.FK = math.sqrt((FKx * FKx) + (FKy * FKy))

    forces.MKx = -zRK * ((FSKy + FAKy) * math.cos(phi) + FSKx * math.sin(phi)) - (zRK - lSK) * Fwky
    forces.MKy = zRK * FKx

    MKx = forces.MKx
    MKy = forces.MKy
    forces.MK = math.sqrt((MKx * MKx) + (MKy * MKy))

    forces.FSKx = FSKx
    forces.FSKy = FSKy
    forces.FAKz = FAKz
    forces.FAKy = FAKy
    forces.Fwky = Fwky
    forces.Fwkz = Fwkz
    forces.FAK_inclined = FAK_inclined

    return forces


def calculate_forces(pdc_file_path, speed_list, zeta_angles, output_file_path):
    pdc_df = pd.read_csv(pdc_file_path)
    pdc_values = pdc_df['PDC'].values

    rB = 42.08
    r_piston = 9.75
    AreaK = np.pi * r_piston * r_piston

    all_results = []

    for speed_rpm in speed_list:
        for zeta_deg in zeta_angles:
            geometry = GeometryPistonGap()
            operating = OperatingPistonGap(speed_rpm=speed_rpm)
            forces = ForcesPistonGap()
            my_input = MyInput()
            my_input.zeta_deg = zeta_deg

            for phi_deg in range(360):
                pDC = pdc_values[phi_deg]
                forces_result = piston_calc_external_forces_inclined_piston(
                    phi_deg, pDC, geometry, operating, forces, my_input, AreaK, rB
                )

                result = {
                    'speed_rpm': speed_rpm,
                    'zeta_deg': zeta_deg,
                    'phi_deg': phi_deg,
                    'pDC': pDC,
                    'FDK': forces_result.FDK,
                    'FaK': forces_result.FaK,
                    'Fwk': forces_result.Fwk,
                    'FKx': forces_result.FKx,
                    'FKy': forces_result.FKy,
                    'FK': forces_result.FK,
                    'MKx': forces_result.MKx,
                    'MKy': forces_result.MKy,
                    'MK': forces_result.MK,
                    'FSKx': forces_result.FSKx,
                    'FSKy': forces_result.FSKy,
                    'FAKz': forces_result.FAKz,
                    'FAKy': forces_result.FAKy,
                    'Fwky': forces_result.Fwky,
                    'Fwkz': forces_result.Fwkz,
                    'FAK_inclined': forces_result.FAK_inclined
                }
                all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file_path, index=False)
    return results_df


def plot_2d_ratio(csv_file_path, output_plot_path=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Calculate mean values for each speed-gamma combination
    grouped = df.groupby(['speed_rpm', 'zeta_deg']).agg({
        'FAKy': 'mean',
        'Fwkz': 'mean'
    }).reset_index()

    # Calculate the REVERSED ratio Fwkz/FAKy (handle division by zero)
    grouped['ratio'] = np.where(grouped['FAKy'] != 0,
                                grouped['Fwkz'] / grouped['FAKy'],
                                0)

    # Get unique values for meshgrid
    speeds = sorted(grouped['speed_rpm'].unique())
    gammas = sorted(grouped['zeta_deg'].unique())

    # Create meshgrid
    Speed, Gamma = np.meshgrid(speeds, gammas)

    # Create ratio matrix
    Ratio = np.zeros_like(Speed, dtype=float)

    for i, gamma in enumerate(gammas):
        for j, speed in enumerate(speeds):
            ratio_val = grouped[(grouped['speed_rpm'] == speed) &
                                (grouped['zeta_deg'] == gamma)]['ratio'].values
            if len(ratio_val) > 0:
                Ratio[i, j] = ratio_val[0]

    # Create enhanced 2D contour plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    # Create filled contour plot
    contourf = ax.contourf(Speed, Gamma, Ratio, levels=20, cmap='RdYlBu_r', alpha=0.8)

    # Add contour lines for better readability
    contours = ax.contour(Speed, Gamma, Ratio, levels=10,
                          colors='black', alpha=0.6, linewidths=1.0)

    # Add contour labels
    ax.clabel(contours, inline=True, fontsize=12, fmt='%.3f',
              inline_spacing=5, manual=False)

    # Enhanced labels with units and engineering context
    ax.set_xlabel('Rotational Speed [RPM] - Pump Operating Speed', fontsize=18, fontweight='bold')
    ax.set_ylabel('Piston Inclination Angle [°]', fontsize=18, fontweight='bold')

    # More descriptive title with engineering significance
    ax.set_title('Piston Force Analysis: Centrifugal-to-Radial Force Ratio\n' +
                 'Fwkz (Centrifugal Force Component) / FAKy (Radial Force)\n' +
                 'Mean Values Across Full Rotation (φ = 0-360°)\n'+
                 'Pressure = 320 bar, Displacement = 100%',
                 fontsize=20, fontweight='bold', pad=25)

    # Enhanced colorbar with better positioning and labels
    cbar = fig.colorbar(contourf, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Force Ratio: Fwkz/FAKy [-]\n(Higher = More Centrifugal Dominant)',
                   fontsize=16, fontweight='bold', labelpad=20)
    cbar.ax.tick_params(labelsize=14)

    # Set axis ticks to show only actual data points
    ax.set_xticks(speeds)
    ax.set_yticks(gammas)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add statistical annotations
    ratio_min = np.min(Ratio)
    ratio_max = np.max(Ratio)
    ratio_mean = np.mean(Ratio)

    # # Add text box with key statistics
    # textstr = (f'Statistics:\n'
    #            f'Min Ratio: {ratio_min:.3f}\n'
    #            f'Max Ratio: {ratio_max:.3f}\n'
    #            f'Mean Ratio: {ratio_mean:.3f}\n'
    #            f'Range: {ratio_max - ratio_min:.3f}')
    # props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, edgecolor='black')
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
    #         verticalalignment='top', bbox=props, fontweight='bold')

    # Mark critical points (max and min ratio locations)
    max_idx = np.unravel_index(np.argmax(Ratio), Ratio.shape)
    min_idx = np.unravel_index(np.argmin(Ratio), Ratio.shape)

    max_speed = Speed[max_idx]
    max_gamma = Gamma[max_idx]
    max_ratio = Ratio[max_idx]

    min_speed = Speed[min_idx]
    min_gamma = Gamma[min_idx]
    min_ratio = Ratio[min_idx]

    # Add markers for critical points
    ax.scatter([max_speed], [max_gamma], color='red', s=200, alpha=1.0,
               marker='*', edgecolors='black', linewidth=2,
               label=f'Max Ratio: {max_ratio:.3f}\n({max_speed:.0f} RPM, {max_gamma:.0f}°)', zorder=5)
    ax.scatter([min_speed], [min_gamma], color='blue', s=200, alpha=1.0,
               marker='*', edgecolors='black', linewidth=2,
               label=f'Min Ratio: {min_ratio:.3f}\n({min_speed:.0f} RPM, {min_gamma:.0f}°)', zorder=5)

    # Add legend for critical points
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=13,
              framealpha=0.9, edgecolor='black')

    # Set tick parameters for better readability
    ax.tick_params(axis='x', labelsize=15, pad=5)
    ax.tick_params(axis='y', labelsize=15, pad=5)

    # Add additional annotations for engineering insight
    engineering_text = ('Insights:\n'
                        '• Red zones: High centrifugal dominance\n'
                        '• Blue zones: High radial dominance\n')
    ax.text(0.02, 0.35, engineering_text, transform=ax.transAxes, fontsize=13,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy'),
            fontweight='normal')

    # Tight layout for better spacing
    plt.tight_layout()

    # Save plot if path provided
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight', facecolor='none', transparent=True)
        plt.savefig(output_plot_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='none', transparent=True)

    plt.show()

    # Print summary statistics for engineering insight
    print("\n=== ENGINEERING ANALYSIS SUMMARY ===")
    print(f"Force Ratio Range: {ratio_min:.3f} to {ratio_max:.3f}")
    print(f"Maximum Ratio occurs at: {max_speed:.0f} RPM, γ = {max_gamma:.1f}°")
    print(f"Minimum Ratio occurs at: {min_speed:.0f} RPM, γ = {min_gamma:.1f}°")
    print(f"Ratio Variation: {((ratio_max - ratio_min) / ratio_mean * 100):.1f}% of mean value")

    # Engineering interpretation
    if ratio_mean > 0:
        print("\nEngineering Insight:")
        print("- Positive ratios indicate centrifugal forces (Fwkz) dominate over radial (FAKy)")
        print("- Higher ratios suggest more centrifugal loading relative to radial forces")
        print("- Consider these ratios for dynamic balancing and bearing design")

    return grouped


def create_2d_animation(csv_file_path, output_animation_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import cm

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Get unique phi angles (for animation frames)
    phi_angles = sorted(df['phi_deg'].unique())
    speeds = sorted(df['speed_rpm'].unique())
    gammas = sorted(df['zeta_deg'].unique())

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 9))

    # Create meshgrid for speed and gamma
    Speed, Gamma = np.meshgrid(speeds, gammas)

    def animate(frame):
        ax.clear()

        # Get current phi angle
        current_phi = phi_angles[frame]

        # Filter data for current phi angle
        phi_data = df[df['phi_deg'] == current_phi]

        # Calculate ratio for current phi
        phi_grouped = phi_data.groupby(['speed_rpm', 'zeta_deg']).agg({
            'FAKy': 'mean',
            'Fwkz': 'mean'
        }).reset_index()

        phi_grouped['ratio'] = np.where(phi_grouped['FAKy'] != 0,
                                        phi_grouped['Fwkz'] / phi_grouped['FAKy'],
                                        0)

        # Create ratio matrix for current phi
        Ratio = np.zeros_like(Speed, dtype=float)

        for i, gamma in enumerate(gammas):
            for j, speed in enumerate(speeds):
                ratio_val = phi_grouped[(phi_grouped['speed_rpm'] == speed) &
                                        (phi_grouped['zeta_deg'] == gamma)]['ratio'].values
                if len(ratio_val) > 0:
                    Ratio[i, j] = ratio_val[0]

        # Create contour plot for current frame
        contourf = ax.contourf(Speed, Gamma, Ratio, levels=20, cmap='RdYlBu_r', alpha=0.8)
        contours = ax.contour(Speed, Gamma, Ratio, levels=10, colors='black', alpha=0.6, linewidths=0.8)

        # Labels and title
        ax.set_xlabel('Rotational Speed [RPM]', fontsize=16, fontweight='bold')
        ax.set_ylabel('Piston Inclination Angle [°]', fontsize=16, fontweight='bold')
        ax.set_title(f'Force Ratio Animation: Fwkz/FAKy\nPhi Angle = {current_phi}°,',
                     fontsize=18, fontweight='bold')

        # Set ticks
        ax.set_xticks(speeds)
        ax.set_yticks(gammas)
        ax.grid(True, alpha=0.3)

        # Add frame info
        frame_text = f'Frame: {frame + 1}/{len(phi_angles)}\nPhi: {current_phi}°'
        ax.text(0.02, 0.98, frame_text, transform=ax.transAxes, fontsize=13,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        return contourf,

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(phi_angles),
                                   interval=200, blit=False, repeat=True)

    # Add colorbar (static)
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r')
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Force Ratio: Fwkz/FAKy [-]', fontsize=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()

    # Save animation if path provided
    if output_animation_path:
        if output_animation_path.endswith('.gif'):
            anim.save(output_animation_path, writer='pillow', fps=5, dpi=150,
                      savefig_kwargs={'facecolor': 'none', 'transparent': True})
        elif output_animation_path.endswith('.mp4'):
            anim.save(output_animation_path, writer='ffmpeg', fps=5, dpi=150,
                      savefig_kwargs={'facecolor': 'none', 'transparent': True})
        print(f"Animation saved as: {output_animation_path}")

    plt.show()

    return anim


# Usage example:
if __name__ == "__main__":
    speed_list = [ 500, 1000, 1500, 2000,2500, 3000,3500,4000, 4500]
    zeta_angles = [0,2,4,6,8,10]

    # Calculate forces and save to CSV
    results = calculate_forces(
        '../Results/P_DC_subset.csv',
        speed_list,
        zeta_angles,
        'piston_forces_results.csv'
    )

    # Create 2D contour plot (static)
    ratio_data = plot_2d_ratio('piston_forces_results.csv', '2d_ratio_plot.png')

    # Create 2D animation/film
    animation_2d = create_2d_animation('piston_forces_results.csv', '2d_ratio_animation.gif')