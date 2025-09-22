import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to read operating conditions
def get_opcons(filepath, start_line, end_line):
    df = pd.read_csv(filepath, sep='\t', skiprows=start_line, nrows=end_line - start_line, header=None)
    df.columns = ['Name', 'Value']
    return df.set_index('Name').to_dict()['Value']

# Function to read piston data
def read_piston(filepath):
    return pd.read_csv(filepath, sep='\t')

# Function to read slipper data
def read_slipper(filepath):
    return pd.read_csv(filepath, sep='\t')

# Set paths and parameters
GlobalPath = "G:/Inline/Highspeed_pump/HP_VD/SIM/RUN_1_Straight_cordinate_piston/simulation_inclined_test"
SavePath = "New_folder"

V = 40  # displacement in cc
maxBeta = 14  # normalization factor for beta
clearance = 20.25  # micrometers
LF = 41.13 * 1000  # Convert bushing length to micrometers

# User input
analysis_choice = input("Enter analysis choice: (p) for piston, (s) for slipper, (b) for both: ").lower()

# Initialize summary dataframe
summary_columns = ['Folder', 'MaxRevLeak', 'TorqueLoss', 'PowerLoss', 'ContactForce',
                   'MaxEccDC', 'MaxEccSP', 'TotalEff', 'HmEff', 'VolEff', 'TiltX', 'TiltY']
Summary = pd.DataFrame(columns=summary_columns)

# Loop over simulation folders
for folder in os.listdir(GlobalPath):
    folder_path = os.path.join(GlobalPath, folder)
    if not os.path.isdir(folder_path):
        continue

    piston_path = os.path.join(folder_path, 'output', 'piston', 'piston.txt')
    slipper_path = os.path.join(folder_path, 'output', 'slipper', 'slipper.txt')
    opcon_path = os.path.join(folder_path, 'input', 'operatingconditions.txt')

    if not os.path.exists(opcon_path):
        continue

    opcon = get_opcons(opcon_path, 10, 28)
    speed = float(opcon.get('speed', 0))
    beta_val = float(opcon.get('beta', 0)) / maxBeta
    HP = float(opcon.get('HP', 0))
    LP = float(opcon.get('LP', 0))

    # Initialize analysis variables
    maxLeak, torqueLoss, powerLoss, contactForce = [np.nan] * 4
    maxEcc_DC, maxEcc_SP, tilt_x, tilt_y = [np.nan] * 4

    if analysis_choice in ['p', 'b'] and os.path.exists(piston_path):
        piston = read_piston(piston_path)
        power_loss = piston['Total_Volumetric_Power_Loss'] + piston['Total_Mechanical_Power_Loss']
        contact_force = piston[['F_Contact_1', 'F_Contact_2', 'F_Contact_3', 'F_Contact_4']].abs().sum(axis=1)
        maxLeak = piston['Total_Leakage'].max() * 60000  # Convert to l/min
        torqueLoss = piston['Total_Torque_Loss'].max()
        powerLoss = power_loss.max()
        contactForce = contact_force.max()
        maxEcc_DC = np.sqrt(piston['e_1']**2 + piston['e_2']**2).max() * 1e6
        maxEcc_SP = np.sqrt(piston['e_3']**2 + piston['e_4']**2).max() * 1e6
        tilt_x = np.degrees(np.arctan((piston['e_1'] - piston['e_3']) / LF)).max()
        tilt_y = np.degrees(np.arctan((piston['e_2'] - piston['e_4']) / LF)).max()

    if analysis_choice in ['s', 'b'] and os.path.exists(slipper_path):
        slipper = read_slipper(slipper_path)
        maxLeak = slipper['QSG_total'].max() * 60000
        torqueLoss = slipper['Powerloss_total'].max()
        contactForce = slipper['avg_p_contact'].max()

    Flow_theo = speed * V * beta_val / 1000
    Flow = Flow_theo - maxLeak
    Power_theo = (HP - LP) * Flow_theo / 600
    PowerCalc = Power_theo - powerLoss / 1000
    eta_vol = (Flow / Flow_theo) * 100 if Flow_theo else np.nan
    eta_ges = (PowerCalc / Power_theo) * 100 if Power_theo else np.nan
    hm_eff = (eta_ges / eta_vol) * 100 if eta_vol else np.nan

    new_row = pd.DataFrame([[folder, maxLeak, torqueLoss, powerLoss, contactForce,
                              maxEcc_DC, maxEcc_SP, eta_ges, hm_eff, eta_vol, tilt_x, tilt_y]],
                            columns=summary_columns)
    Summary = pd.concat([Summary, new_row], ignore_index=True)

# Save results
Summary.to_csv(os.path.join(SavePath, 'Results.csv'), index=False)

# Plot summary data
fig, axs = plt.subplots(4, 3, figsize=(12, 10))
axs = axs.ravel()
plot_titles = ['MaxRev Leakage', 'Torque Loss', 'Power Loss', 'Contact Force', 'Max Ecc DC', 'Max Ecc SP',
               'Total Efficiency', 'HM Efficiency', 'Volumetric Efficiency', 'Tilt (Radial, x)', 'Tilt (Axial, y)']
y_labels = ['Leakage [l/min]', 'Torque Loss', 'Power Loss', 'Force', 'DC [\u03bcm]', 'SP [\u03bcm]',
            'Efficiency (%)', 'Efficiency (%)', 'Efficiency (%)', 'Angle [°]', 'Angle [°]']

for i, col in enumerate(summary_columns[1:]):
    axs[i].plot(Summary['Folder'], Summary[col], marker='o')
    axs[i].set_title(plot_titles[i])
    axs[i].set_xlabel('Folder')
    axs[i].set_ylabel(y_labels[i])
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SavePath, 'SummaryPlots.png'))
plt.show()
