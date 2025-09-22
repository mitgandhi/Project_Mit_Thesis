# analyze_simulation.py
"""
This script processes simulation results for piston (Kolben) and slipper.
Block analysis is removed so that only piston and slipper data are considered.
Use the user input to choose to analyze:
  (p) Piston only
  (s) Slipper only
  (b) Both piston and slipper

Make sure the helper functions (read_piston, read_slipper, get_opcons) are available.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import math

# Set default font for all figures
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 18


def read_header_piston(file_path):
    """Read the header from the piston file."""
    with open(file_path, 'r') as file:
        header_line = file.readline().strip()

    # Split the header into column names
    header = header_line.split('\t')
    return header


def read_piston_s(file_path):
    """Read the piston data from the file."""
    header = read_header_piston(file_path)
    data = pd.read_csv(file_path, delimiter='\t', skiprows=1, names=header)
    return data


def read_slipper(file_path):
    """Read the slipper data from the file."""
    with open(file_path, 'r') as file:
        header_line = file.readline().strip()

    # Split the header into column names
    header = header_line.split('\t')
    data = pd.read_csv(file_path, delimiter='\t', skiprows=1, names=header)
    return data, header


def get_opcons(file_path, start_line, end_line):
    """Read operating conditions from the file."""
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[start_line - 1:end_line]
        for line in lines:
            parts = line.strip().split(':')
            if len(parts) >= 2:
                name = parts[0].strip()
                value = parts[1].strip()
                data.append({'Name': name, 'Value': value})

    return pd.DataFrame(data)


def main():
    plt.close('all')

    # User chooses analysis type
    analysis_choice = input('Enter analysis choice: (p) for piston, (s) for slipper, (b) for both: ').lower()

    # Set simulation paths and parameters
    first_run = 1  # used for saving/loading if needed
    server = 0  # set to 1 if using server paths
    mac = 0  # change to 1 if running on macOS

    if mac == 1:
        global_path = '/Volumes/Porsche/Simulations/LMB/Standard/WithHeat_Piston_noDef'
        sep = '/'
    else:
        global_path = r'G:\Inline\Highspeed_pump\HP_VD\SIM\RUN_1_Straight_cordinate_piston\simulation_inclined_test\output\piston'
        sep = '\\'
        if server:
            input_path = r'Z:\ServerSim\02Inline\Run7'

    save_path = 'save_path'
    os.makedirs(save_path, exist_ok=True)

    run_once = 1
    path_info = os.listdir(global_path)
    V = 40  # displacement in cc
    max_beta = 12  # normalization factor for beta

    # New variable: clearance (in micrometers) for bore boundaries
    clearance = 20.25  # nominal clearance in [μm]

    # New variable: bushing length LF in mm, and in micrometers
    LF = 41.13  # bushing length in mm
    LF_um = LF * 1000  # convert LF to micrometers

    # Initialize summary table
    summary = pd.DataFrame(columns=[
        'Folder', 'MaxRevLeak', 'TorqueLoss', 'PowerLoss', 'ContactForce',
        'MaxEccDC', 'MaxEccSP', 'TotalEff', 'HmEff', 'VolEff', 'TiltX', 'TiltY'
    ])

    # Loop over simulation folders
    for folder in path_info:
        if folder in ['.', '..']:
            continue

        sim_path = os.path.join(global_path, folder)

        # Skip folders that are not valid simulation folders
        try:
            # Check if folder name starts with letter and followed by numbers
            if not (folder[0].isalpha() and folder[1:3].isdigit()):
                if 'Results.txt' in folder or 'Thumbs.db' in folder or '.png' in folder:
                    continue
                else:
                    print(f'{folder} is not a valid simulation folder.')
                    continue
        except (IndexError, ValueError):
            print(f'{folder} is not a valid simulation folder.')
            continue

        # Construct file paths (only piston and slipper remain)
        if server:
            input_path_folder = os.path.join(input_path, folder)
            piston_path = os.path.join(sim_path, 'piston', 'piston.txt')
            slipper_path = os.path.join(sim_path, 'slipper', 'slipper.txt')
            pressure_path = os.path.join(sim_path, 'pressure', 'pressure_results.txt')
            opcon_path = os.path.join(input_path_folder, 'input', 'operatingconditions.txt')
        else:
            piston_path = os.path.join(sim_path, 'output', 'piston', 'piston.txt')
            slipper_path = os.path.join(sim_path, 'output', 'slipper', 'slipper.txt')
            pressure_path = os.path.join(sim_path, 'output', 'pressure', 'pressure_results.txt')
            opcon_path = os.path.join(sim_path, 'input', 'operatingconditions.txt')

        # Check if files exist
        if analysis_choice in ['p', 'b'] and not os.path.exists(piston_path):
            print(f'Piston file not found in {folder}')
            continue

        if analysis_choice in ['s', 'b'] and not os.path.exists(slipper_path):
            print(f'Slipper file not found in {folder}')
            continue

        # Read operating conditions (used in both cases)
        try:
            operating_conditions = get_opcons(opcon_path, 10, 28)
            speed = float(operating_conditions[operating_conditions['Name'] == 'speed']['Value'].values[0])
            beta_val = float(operating_conditions[operating_conditions['Name'] == 'beta']['Value'].values[0]) / max_beta
            HP = float(operating_conditions[operating_conditions['Name'] == 'HP']['Value'].values[0])
            LP = float(operating_conditions[operating_conditions['Name'] == 'LP']['Value'].values[0])
        except Exception as e:
            print(f'Error reading operating conditions from {folder}: {e}')
            continue

        # Initialize summary variables
        Leak = np.nan
        PLoss = np.nan

        # Process piston data if chosen (p or b)
        if analysis_choice in ['p', 'b']:
            try:
                piston = read_piston_s(piston_path)
                header_piston = read_header_piston(piston_path)

                # Extract piston signals
                t_p = piston['%time'] if '%time' in piston.columns else piston['time']
                rev_p = piston['revolution']
                phi_p = piston['shaft_angle']
                Leakage_p = piston['Total_Leakage'] * 60000  # convert to l/min

                # Find the columns with power loss
                vol_power_cols = [col for col in piston.columns if 'Total_Volumetric_Power_Loss' in col]
                Power_L = piston[vol_power_cols[0]] if vol_power_cols else np.zeros(len(piston))

                mech_power_cols = [col for col in piston.columns if 'Total_Mechanical_Power_Loss' in col]
                Power_F = piston[mech_power_cols[0]] if mech_power_cols else np.zeros(len(piston))

                Power_p = Power_L + Power_F
                P_f_p = piston['Total_Torque_Loss']

                # Eccentricity signals (convert from m to micrometers)
                e1 = piston['e_1'] * 1e6
                e2 = piston['e_2'] * 1e6
                e3 = piston['e_3'] * 1e6
                e4 = piston['e_4'] * 1e6

                # Contact forces (sum of absolute values from 4 sensors)
                cp1 = piston['F_Contact_1']
                cp2 = piston['F_Contact_2']
                cp3 = piston['F_Contact_3']
                cp4 = piston['F_Contact_4']
                contact_p = abs(cp1) + abs(cp2) + abs(cp3) + abs(cp4)

                # Determine indices for the final (or last complete) revolution
                max_rev_p = round(max(rev_p))
                n1_p = np.where(rev_p >= max_rev_p - 1)[0][0] + 1
                n2_p = len(rev_p) - 2

                # Plot Piston Data
                plt.figure(figsize=(8, 6))
                plt.plot(phi_p[n1_p:n2_p], Power_p[n1_p:n2_p], 'k', linewidth=2)
                plt.hold = True
                plt.plot(phi_p[n1_p:n2_p], Power_F[n1_p:n2_p], 'r', linewidth=2)
                plt.plot(phi_p[n1_p:n2_p], Power_L[n1_p:n2_p], 'b', linewidth=2)
                plt.title(f'Piston Powerloss - {speed} rpm, ΔP:{round(HP - LP)} bar, β:{beta_val * 100} %')
                plt.xlabel('Shaft Rotation [°]')
                plt.ylabel('Powerloss [W]')
                plt.grid(True)
                plt.legend(['Total', 'Friction', 'Leakage'])
                plt.xlim([0, 360])
                plt.xticks(np.arange(0, 361, 60))
                plt.savefig(os.path.join(sim_path, 'PistonLeakage.png'))
                plt.close()

                plt.figure(figsize=(8, 6))
                plt.plot(phi_p[n1_p:n2_p], Power_p[n1_p:n2_p], 'k', linewidth=2)
                plt.title(
                    f'Piston Power Loss - {int(speed)} rpm, ΔP: {int(round(HP - LP))} bar, β: {int(beta_val * 100)}%')
                plt.xlabel('Shaft Rotation [°]')
                plt.ylabel('Power Loss [W]')
                plt.grid(True)
                plt.xlim([0, 360])
                plt.xticks(np.arange(0, 361, 60))
                plt.savefig(os.path.join(sim_path, 'PistonPowerLoss.png'))
                plt.close()

                plt.figure(figsize=(8, 6))
                plt.plot(phi_p[n1_p:n2_p], P_f_p[n1_p:n2_p], 'k', linewidth=2)
                plt.title(
                    f'Piston Friction Loss - {int(speed)} rpm, ΔP: {int(round(HP - LP))} bar, β: {int(beta_val * 100)}%')
                plt.xlabel('Shaft Rotation [°]')
                plt.ylabel('Friction Loss [W]')
                plt.grid(True)
                plt.xlim([0, 360])
                plt.xticks(np.arange(0, 361, 60))
                plt.savefig(os.path.join(sim_path, 'PistonFrictionLoss.png'))
                plt.close()

                plt.figure(figsize=(8, 6))
                plt.plot(phi_p[n1_p:n2_p], contact_p[n1_p:n2_p], 'k', linewidth=2)
                plt.title(
                    f'Piston Contact Force - {int(speed)} rpm, ΔP: {int(round(HP - LP))} bar, β: {int(beta_val * 100)}%')
                plt.xlabel('Shaft Rotation [°]')
                plt.ylabel('Contact Force [N]')
                plt.grid(True)
                plt.xlim([0, 360])
                plt.xticks(np.arange(0, 361, 60))
                plt.savefig(os.path.join(sim_path, 'PistonContact.png'))
                plt.close()

                # New Feature 1: Plot the eccentricity signals vs. shaft angle
                plt.figure(figsize=(8, 6))
                plt.plot(phi_p[n1_p:n2_p], e1[n1_p:n2_p], 'b', linewidth=1.5)
                plt.plot(phi_p[n1_p:n2_p], e2[n1_p:n2_p], 'r', linewidth=1.5)
                plt.plot(phi_p[n1_p:n2_p], e3[n1_p:n2_p], 'g', linewidth=1.5)
                plt.plot(phi_p[n1_p:n2_p], e4[n1_p:n2_p], 'm', linewidth=1.5)
                plt.title('Eccentricity Signals (in μm)')
                plt.xlabel('Shaft Rotation [°]')
                plt.ylabel('Eccentricity [μm]')
                plt.legend(['e1', 'e2', 'e3', 'e4'])
                plt.grid(True)
                plt.xlim([0, 360])
                plt.xticks(np.arange(0, 361, 60))
                plt.savefig(os.path.join(sim_path, 'EccentricitySignals.png'))
                plt.close()

                # New Feature 2: Polar plot for eccentricity
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

                # DC Side (previously "piston front")
                for idx in range(n1_p, n2_p):
                    marker_size = 180 - (phi_p.iloc[idx] - phi_p.iloc[n1_p]) * 1
                    if marker_size < 5:
                        marker_size = 5
                    ax1.scatter(e1.iloc[idx], e2.iloc[idx], s=marker_size, c='b', alpha=0.8)

                # Draw bold circle representing the bore (radius = clearance)
                t = np.linspace(0, 2 * np.pi, 100)
                x_circ = clearance * np.cos(t)
                y_circ = clearance * np.sin(t)
                ax1.plot(x_circ, y_circ, 'k', linewidth=2)
                ax1.set_title('Piston DC Side (e1 vs e2)')
                ax1.set_xlabel('e1 [μm]')
                ax1.set_ylabel('e2 [μm]')
                ax1.grid(True)
                ax1.axis('equal')
                ax1.set_xlim([-1.2 * clearance, 1.2 * clearance])
                ax1.set_ylim([-1.2 * clearance, 1.2 * clearance])

                # SP Side (previously "piston back")
                for idx in range(n1_p, n2_p):
                    marker_size = 180 - (phi_p.iloc[idx] - phi_p.iloc[n1_p]) * 1
                    if marker_size < 5:
                        marker_size = 5
                    ax2.scatter(e3.iloc[idx], e4.iloc[idx], s=marker_size, c='r', alpha=0.8)

                # Draw bold circle representing the bore (radius = clearance)
                ax2.plot(x_circ, y_circ, 'k', linewidth=2)
                ax2.set_title('Piston SP Side (e3 vs e4)')
                ax2.set_xlabel('e3 [μm]')
                ax2.set_ylabel('e4 [μm]')
                ax2.grid(True)
                ax2.axis('equal')
                ax2.set_xlim([-1.2 * clearance, 1.2 * clearance])
                ax2.set_ylim([-1.2 * clearance, 1.2 * clearance])

                plt.tight_layout()
                plt.savefig(os.path.join(sim_path, 'PolarEccentricity.png'))
                plt.close()

                # New Feature 4: Calculate piston tilt angles using LF
                # Compute mean center positions over the final revolution for DC and SP sides:
                DC_x = np.nanmean(e1[n1_p:n2_p])
                DC_y = np.nanmean(e2[n1_p:n2_p])
                SP_x = np.nanmean(e3[n1_p:n2_p])
                SP_y = np.nanmean(e4[n1_p:n2_p])

                # Tilt (radial, x) and tilt (axial, y) in degrees:
                tilt_x = np.arctan((DC_x - SP_x) / LF_um) * (180 / np.pi)
                tilt_y = np.arctan((DC_y - SP_y) / LF_um) * (180 / np.pi)

                tilt_x_full = np.arctan((e1 - e3) / LF_um) * (180 / np.pi)
                tilt_y_full = np.arctan((e2 - e4) / LF_um) * (180 / np.pi)

                plt.figure(figsize=(8, 6))
                plt.plot(phi_p[n1_p:n2_p], tilt_x_full[n1_p:n2_p], 'r', linewidth=2)
                plt.plot(phi_p[n1_p:n2_p], tilt_y_full[n1_p:n2_p], 'b', linewidth=2)
                plt.title(f'Piston tilt - {int(speed)} rpm, ΔP: {int(round(HP - LP))} bar, β: {int(beta_val * 100)}%')
                plt.xlabel('Shaft Rotation [°]')
                plt.ylabel('Tilt Angle [deg]')
                plt.grid(True)
                plt.legend(['X Tilt', 'Y Tilt'])
                plt.xlim([0, 360])
                plt.xticks(np.arange(0, 361, 60))
                plt.savefig(os.path.join(sim_path, 'PistonTilt.png'))
                plt.close()

            except Exception as e:
                print(f'Error processing piston data for {folder}: {e}')
                continue

        # Process slipper data if chosen (s or b)
        if analysis_choice in ['s', 'b']:
            try:
                slipper, header_slipper = read_slipper(slipper_path)

                # Extract slipper signals
                t_s = slipper['%t'] if '%t' in slipper else slipper['t']
                rev_s = slipper['rev']
                phi_s = slipper['phi']
                Leakage_s = slipper['QSG_total'] * 60000  # convert to l/min

                # Friction loss is computed as total power loss minus volumetric leakage loss
                P_L_s = slipper['Total_Volumetric_Power_Loss']
                P_f_s = slipper['Powerloss_total'] - P_L_s

                # Average contact pressure of the slipper
                contact_s = slipper['avg_p_contact']

                # Determine indices for the final revolution in slipper data
                max_rev_s = round(max(rev_s))
                n1_s = np.where(rev_s >= max_rev_s - 1)[0][0]
                n2_s = len(rev_s)

                plt.figure(figsize=(8, 6))
                plt.plot(phi_s[n1_s:n2_s], Leakage_s[n1_s:n2_s], 'r', linewidth=2)
                plt.title(
                    f'Slipper Leakage - {int(speed)} rpm, ΔP: {int(round(HP - LP))} bar, β: {int(beta_val * 100)}%')
                plt.xlabel('Shaft Rotation [°]')
                plt.ylabel('Leakage [l/min]')
                plt.grid(True)
                plt.legend(['Slipper'])
                plt.savefig(os.path.join(sim_path, 'SlipperLeakage.png'))
                plt.close()

                plt.figure(figsize=(8, 6))
                plt.plot(phi_s[n1_s:n2_s], P_f_s[n1_s:n2_s], 'r', linewidth=2)
                plt.title(
                    f'Slipper Friction Loss - {int(speed)} rpm, ΔP: {int(round(HP - LP))} bar, β: {int(beta_val * 100)}%')
                plt.xlabel('Shaft Rotation [°]')
                plt.ylabel('Friction Loss [W]')
                plt.grid(True)
                plt.legend(['Slipper'])
                plt.savefig(os.path.join(sim_path, 'SlipperFrictionLoss.png'))
                plt.close()

                plt.figure(figsize=(8, 6))
                plt.plot(phi_s[n1_s:n2_s], contact_s[n1_s:n2_s], 'r', linewidth=2)
                plt.title(
                    f'Slipper Contact Pressure - {int(speed)} rpm, ΔP: {int(round(HP - LP))} bar, β: {int(beta_val * 100)}%')
                plt.xlabel('Shaft Rotation [°]')
                plt.ylabel('Contact Pressure [bar]')
                plt.grid(True)
                plt.legend(['Slipper'])
                plt.savefig(os.path.join(sim_path, 'SlipperContact.png'))
                plt.close()

            except Exception as e:
                print(f'Error processing slipper data for {folder}: {e}')
                continue

        # Summary calculations and results saving
        # Theoretical flow (l/min)
        Flow_theo = speed * V * beta_val / 1000

        if analysis_choice == 'p':
            Leak = np.nanmean(Leakage_p[n1_p:n2_p])
            PLoss = np.nanmean(P_f_p[n1_p:n2_p])
            maxLeak = np.max(Leakage_p[n1_p:n2_p])
            torqueLoss = np.max(P_f_p[n1_p:n2_p])
            powerLoss = np.max(Power_p[n1_p:n2_p])
            contactForce = np.max(contact_p[n1_p:n2_p])
            maxEcc_DC = np.max(np.sqrt(e1[n1_p:n2_p] ** 2 + e2[n1_p:n2_p] ** 2))
            maxEcc_SP = np.max(np.sqrt(e3[n1_p:n2_p] ** 2 + e4[n1_p:n2_p] ** 2))
        elif analysis_choice == 's':
            Leak = np.nanmean(Leakage_s[n1_s:n2_s])
            PLoss = np.nanmean(P_f_s[n1_s:n2_s])
            maxLeak = np.max(Leakage_s[n1_s:n2_s])
            torqueLoss = np.max(P_f_s[n1_s:n2_s])
            powerLoss = np.nan
            contactForce = np.max(contact_s[n1_s:n2_s])
            maxEcc_DC = np.nan
            maxEcc_SP = np.nan
            # For slipper, tilt angles are not calculated.
            tilt_x = np.nan
            tilt_y = np.nan
        elif analysis_choice == 'b':
            Leak = np.nanmean(Leakage_p[n1_p:n2_p]) + np.nanmean(Leakage_s[n1_s:n2_s])
            PLoss = np.nanmean(P_f_p[n1_p:n2_p]) + np.nanmean(P_f_s[n1_s:n2_s])
            maxLeak = np.max(Leakage_p[n1_p:n2_p]) + np.max(Leakage_s[n1_s:n2_s])
            torqueLoss = np.max(P_f_p[n1_p:n2_p]) + np.max(P_f_s[n1_s:n2_s])
            powerLoss = np.max(Power_p[n1_p:n2_p])
            contactForce = np.max(contact_p[n1_p:n2_p]) + np.max(contact_s[n1_s:n2_s])
            maxEcc_DC = np.max(np.sqrt(e1[n1_p:n2_p] ** 2 + e2[n1_p:n2_p] ** 2))
            maxEcc_SP = np.max(np.sqrt(e3[n1_p:n2_p] ** 2 + e4[n1_p:n2_p] ** 2))
            # Use tilt computed from piston data

        Flow = Flow_theo - Leak
        Power_theo = (HP - LP) * Flow_theo / 600  # theoretical power in kW
        PowerCalc = Power_theo - PLoss / 1000  # corrected power (kW)
        eta_vol = (Flow / Flow_theo) * 100
        eta_ges = (PowerCalc / Power_theo) * 100
        if eta_vol != 0:
            hm_eff = (eta_ges / eta_vol) * 100
        else:
            hm_eff = np.nan

        # New Feature 3: Print summary information to the command line
        print(f'Folder: {folder}')
        print(f'Rev: {max(rev_p):.1f}')
        print(
            f'Leakage: {maxLeak:.2f}, Torque loss: {torqueLoss:.2f}, Power loss: {powerLoss:.2f}, Contact force: {contactForce:.2f}')
        print(
            f'Max Ecc DC: {maxEcc_DC:.2f}, Max Ecc SP: {maxEcc_SP:.2f}, Total Eff: {eta_ges:.2f}, HM Eff: {hm_eff:.2f}, Vol Eff: {eta_vol:.2f}')
        print(f'Tilt (radial, x): {tilt_x:.2f}, Tilt (axial, y): {tilt_y:.2f}')

        # Save summary data to Results.txt in SavePath
        results_file = os.path.join(save_path, 'Results.txt')
        if os.path.exists(results_file) and folder == path_info[2]:
            with open(results_file, 'w') as fid:
                fid.write('Speed[rpm]\tHP[bar]\tLP[bar]\tbeta\tFlow[l/min]\tLeakage[l/min]\tPowerLoss[W]\n')
        else:
            with open(results_file, 'a') as fid:
                fid.write(
                    f'{speed:.0f}\t{HP:.2f}\t{LP:.2f}\t{beta_val:.2f}\t{Flow:.2f}\t{Leak:.2f}\t{PowerCalc * 1000:.2f}\n')

        # New Feature 4: Append summary values to the table
        new_row = pd.DataFrame({
            'Folder': [folder],
            'MaxRevLeak': [maxLeak],
            'TorqueLoss': [torqueLoss],
            'PowerLoss': [powerLoss],
            'ContactForce': [contactForce],
            'MaxEccDC': [maxEcc_DC],
            'MaxEccSP': [maxEcc_SP],
            'TotalEff': [eta_ges],
            'HmEff': [hm_eff],
            'VolEff': [eta_vol],
            'TiltX': [tilt_x],
            'TiltY': [tilt_y]
        })
        summary = pd.concat([summary, new_row], ignore_index=True)

   # #  New Feature 5: Plot the summary table contents vs. Folder names
    ax2 = plt.subplot(4, 3, 2)
    summary.plot(x='Folder', y='TorqueLoss', marker='o', linestyle='-', ax=ax2)
    ax2.set_title('Torque Loss')
    ax2.set_xlabel('Folder')
    ax2.set_ylabel('Torque Loss [W]')
    ax2.grid(True)

    ax3 = plt.subplot(4, 3, 3)
    summary.plot(x='Folder', y='PowerLoss', marker='o', linestyle='-', ax=ax3)
    ax3.set_title('Power Loss')
    ax3.set_xlabel('Folder')
    ax3.set_ylabel('Power Loss [W]')
    ax3.grid(True)

    ax4 = plt.subplot(4, 3, 4)
    summary.plot(x='Folder', y='ContactForce', marker='o', linestyle='-', ax=ax4)
    ax4.set_title('Contact Force')
    ax4.set_xlabel('Folder')
    ax4.set_ylabel('Contact Force [N]')
    ax4.grid(True)

    ax5 = plt.subplot(4, 3, 5)
    summary.plot(x='Folder', y='MaxEccDC', marker='o', linestyle='-', ax=ax5)
    ax5.set_title('Max Eccentricity DC')
    ax5.set_xlabel('Folder')
    ax5.set_ylabel('Eccentricity [μm]')
    ax5.grid(True)

    ax6 = plt.subplot(4, 3, 6)
    summary.plot(x='Folder', y='MaxEccSP', marker='o', linestyle='-', ax=ax6)
    ax6.set_title('Max Eccentricity SP')
    ax6.set_xlabel('Folder')
    ax6.set_ylabel('Eccentricity [μm]')
    ax6.grid(True)

    ax7 = plt.subplot(4, 3, 7)
    summary.plot(x='Folder', y='TotalEff', marker='o', linestyle='-', ax=ax7)
    ax7.set_title('Total Efficiency')
    ax7.set_xlabel('Folder')
    ax7.set_ylabel('Efficiency [%]')
    ax7.grid(True)

    ax8 = plt.subplot(4, 3, 8)
    summary.plot(x='Folder', y='HmEff', marker='o', linestyle='-', ax=ax8)
    ax8.set_title('Hydromechanical Efficiency')
    ax8.set_xlabel('Folder')
    ax8.set_ylabel('Efficiency [%]')
    ax8.grid(True)

    ax9 = plt.subplot(4, 3, 9)
    summary.plot(x='Folder', y='VolEff', marker='o', linestyle='-', ax=ax9)
    ax9.set_title('Volumetric Efficiency')
    ax9.set_xlabel('Folder')
    ax9.set_ylabel('Efficiency [%]')
    ax9.grid(True)

    ax10 = plt.subplot(4, 3, 10)
    summary.plot(x='Folder', y='TiltX', marker='o', linestyle='-', ax=ax10)
    ax10.set_title('Tilt X (Radial)')
    ax10.set_xlabel('Folder')
    ax10.set_ylabel('Tilt Angle [deg]')
    ax10.grid(True)

    ax11 = plt.subplot(4, 3, 11)
    summary.plot(x='Folder', y='TiltY', marker='o', linestyle='-', ax=ax11)
    ax11.set_title('Tilt Y (Axial)')
    ax11.set_xlabel('Folder')
    ax11.set_ylabel('Tilt Angle [deg]')
    ax11.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'SummaryPlots.png'))
    plt.close()

    # Save summary table to CSV
    summary_file = os.path.join(save_path, 'SummaryResults.csv')
    summary.to_csv(summary_file, index=False)
    print(f'Summary results saved to {summary_file}')

    # Display summary table
    print("\nSummary Table:")
    print(summary)


if __name__ == "__main__":
    main()