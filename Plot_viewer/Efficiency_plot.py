#!/usr/bin/env python3
"""
batch_analyze_piston_multi_folders.py
Batch script to analyze piston behavior & efficiency for multiple base folders
Enhanced with presentation-ready plots for master thesis
Python version - Comparison plots only
Modified to save plots to specified folder with transparent background
UPDATED: Increased font sizes for PowerPoint presentations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# === PLOT CONFIGURATION FOR PRESENTATION ===
# Set matplotlib parameters for high-quality presentation plots with LARGER FONTS
plt.rcParams.update({
    'figure.facecolor': 'none',  # Transparent background
    'axes.facecolor': 'none',  # Transparent axes background
    'font.size': 16,  # Increased from 12 to 16 - Base font size
    'font.family': 'sans-serif',  # Standard font family
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'axes.labelsize': 18,  # Increased from 12 to 18 - X and Y axis labels
    'axes.titlesize': 20,  # Increased from 14 to 20 - Plot titles
    'xtick.labelsize': 16,  # X-axis tick labels
    'ytick.labelsize': 16,  # Y-axis tick labels
    'legend.fontsize': 14,  # Legend font size
    'lines.linewidth': 3.0,  # Increased from 2.5 to 3.0 - Thicker lines
    'axes.linewidth': 2.0,  # Increased from 1.5 to 2.0 - Thicker axes
    'grid.linewidth': 1.2,  # Increased from 1.0 to 1.2 - Thicker grid
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Define presentation colors (colorblind-friendly palette)
presentation_colors = [
    [0.2196, 0.4235, 0.6902],  # Blue
    [0.8431, 0.1882, 0.1529],  # Red
    [0.2157, 0.4941, 0.7216],  # Light Blue
    [0.5961, 0.3059, 0.6392],  # Purple
    [0.1176, 0.5333, 0.8980],  # Cyan
    [0.9412, 0.6314, 0.1294],  # Orange
    [0.4627, 0.7176, 0.2745],  # Green
]

# Specify the output folder for plots
PLOT_OUTPUT_FOLDER = r'Z:\Studenten\Mit\Thesis\PyProject'


def clean_folder_label(label):
    """Clean the folder label for safe file naming"""
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', label)
    cleaned = cleaned.strip()
    return cleaned if cleaned else 'Folder_Unknown'


def read_geometry_data(geom_path):
    """Read dK value from geometry.txt file"""
    dK = np.nan
    if os.path.isfile(geom_path):
        try:
            with open(geom_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('dK'):
                        match = re.search(r'dK\s+([\d\.]+)', line)
                        if match:
                            dK = float(match.group(1))
                            break
        except Exception as e:
            print(f"⚠️ Error reading geometry file: {e}")
    return dK


def read_operating_conditions(op_path):
    """Read operating conditions from operatingconditions.txt file"""
    conditions = {'speed': np.nan, 'beta': np.nan, 'maxBeta': np.nan, 'HP': np.nan, 'LP': np.nan}

    if os.path.isfile(op_path):
        try:
            with open(op_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('speed') and 'betamax' not in line:
                        match = re.search(r'speed\s+([\d\.]+)', line)
                        if match:
                            conditions['speed'] = float(match.group(1))
                        else:
                            try:
                                conditions['speed'] = float(line.split('speed')[-1].strip())
                            except:
                                pass
                    elif line.startswith('beta') and 'max' not in line:
                        match = re.search(r'beta\s+([\d\.]+)', line)
                        if match:
                            conditions['beta'] = float(match.group(1))
                        else:
                            try:
                                conditions['beta'] = float(line.split('beta')[-1].strip())
                            except:
                                pass
                    elif line.startswith('betamax'):
                        match = re.search(r'betamax\s+([\d\.]+)', line)
                        if match:
                            conditions['maxBeta'] = float(match.group(1))
                        else:
                            try:
                                conditions['maxBeta'] = float(line.split('betamax')[-1].strip())
                            except:
                                pass
                    elif line.startswith('HP'):
                        match = re.search(r'HP\s+([\d\.]+)', line)
                        if match:
                            conditions['HP'] = float(match.group(1))
                        else:
                            try:
                                conditions['HP'] = float(line.split('HP')[-1].strip())
                            except:
                                pass
                    elif line.startswith('LP'):
                        match = re.search(r'LP\s+([\d\.]+)', line)
                        if match:
                            conditions['LP'] = float(match.group(1))
                        else:
                            try:
                                conditions['LP'] = float(line.split('LP')[-1].strip())
                            except:
                                pass
        except Exception as e:
            print(f"⚠️ Error reading operating conditions: {e}")

    return conditions


def read_piston_data(piston_file):
    """Read piston data from piston.txt file"""
    try:
        # Try reading with pandas
        piston = pd.read_csv(piston_file, sep='\t')

        # Clean column names (replace % with X_)
        piston.columns = [col.replace('%', 'X_') for col in piston.columns]

        # Define column mappings
        column_mapping = {
            'rev_column': 'revolution',
            'time_column': 'X_time',
            'leakage_column': None,
            'mech_loss_column': None,
            'vol_loss_column': None
        }

        # Find leakage column
        leakage_candidates = ['Total_Leakage', 'Leakage_Total']
        for candidate in leakage_candidates:
            if candidate in piston.columns:
                column_mapping['leakage_column'] = candidate
                break

        # Find mechanical loss column
        mech_loss_candidates = ['Total_Mechanical_Power_Loss', 'Total_Power_Loss', 'Power_Loss_Mechanical']
        for candidate in mech_loss_candidates:
            if candidate in piston.columns:
                column_mapping['mech_loss_column'] = candidate
                break

        # Find volumetric loss column
        vol_loss_candidates = ['Total_Volumetric_Power_Loss', 'Power_Loss_Volumetric']
        for candidate in vol_loss_candidates:
            if candidate in piston.columns:
                column_mapping['vol_loss_column'] = candidate
                break

        return piston, column_mapping

    except Exception as e:
        print(f"⚠️ Error reading piston data: {e}")
        return None, None


def process_folder(base_folder, displacement_volume, folder_label):
    """Process a single base folder and return results"""
    print(f'\n\n===================================================================')
    print(f'PROCESSING BASE FOLDER: {folder_label}')
    print(f'Displacement Volume: {displacement_volume} cc')
    print(f'===================================================================\n')

    results = []
    folder_short_name = os.path.basename(base_folder)

    # Loop over subfolders
    try:
        subfolders = [d for d in os.listdir(base_folder)
                      if os.path.isdir(os.path.join(base_folder, d))
                      and not d.startswith('.')
                      and not d.startswith('Thumbs')
                      and 'Results' not in d
                      and '.png' not in d]
    except Exception as e:
        print(f"⚠️ Error accessing base folder {base_folder}: {e}")
        return []

    for sim_name in subfolders:
        sim_path = os.path.join(base_folder, sim_name)
        piston_file = os.path.join(sim_path, 'output', 'piston', 'piston.txt')

        if not os.path.isfile(piston_file):
            print(f'Skipping {sim_name} — piston.txt not found.')
            continue

        # Read geometry data
        geom_path = os.path.join(sim_path, 'input', 'geometry.txt')
        dK = read_geometry_data(geom_path)
        if np.isnan(dK):
            print(f'⚠️ geometry.txt not found for {sim_name}.')

        # Read operating conditions
        op_path = os.path.join(sim_path, 'input', 'operatingconditions.txt')
        conditions = read_operating_conditions(op_path)

        if any(np.isnan(list(conditions.values()))):
            print(f'⚠️ Incomplete operating conditions in {sim_name} — skipping.')
            continue

        speed, beta, max_beta, HP, LP = conditions.values()
        dp = HP - LP

        # Read piston data
        piston, column_mapping = read_piston_data(piston_file)
        if piston is None or column_mapping['rev_column'] not in piston.columns:
            print(f'⚠️ Error reading piston data for {sim_name} — skipping.')
            continue

        # Rev range for last rotation
        revs = piston[column_mapping['rev_column']]
        max_rev = int(np.floor(revs.max()))

        if max_rev < 1:
            print(f'⚠️ Not enough data for full revolution in {sim_name}.')
            continue

        # Find indices for last revolution
        n1 = np.where(revs >= max_rev - 1)[0][0]
        n2 = np.where(revs >= max_rev)[0][-1]

        # Efficiency calculations
        beta_norm = beta / max_beta
        flow_theo = speed * displacement_volume * beta_norm / 1000  # l/min

        # Get leakage data
        if column_mapping['leakage_column'] and column_mapping['leakage_column'] in piston.columns:
            leakage = piston[column_mapping['leakage_column']].iloc[n1:n2 + 1].mean() * 60000  # l/min
        else:
            leakage = 0
            print(f'⚠️ Leakage column not found in {sim_name} — assuming zero.')

        flow = flow_theo - leakage
        nu_vol = flow / flow_theo

        # Calculate power losses
        if column_mapping['mech_loss_column'] and column_mapping['mech_loss_column'] in piston.columns:
            ploss_f = piston[column_mapping['mech_loss_column']].iloc[n1:n2 + 1].mean()
        else:
            ploss_f = 0
            print(f'⚠️ Mechanical loss column not found in {sim_name} — assuming zero.')

        if column_mapping['vol_loss_column'] and column_mapping['vol_loss_column'] in piston.columns:
            ploss_l = piston[column_mapping['vol_loss_column']].iloc[n1:n2 + 1].mean()
        else:
            ploss_l = leakage * dp / 600 * 1000  # W
            print(f'⚠️ Volumetric loss column not found in {sim_name} — estimating from leakage.')

        ploss = ploss_f + ploss_l

        p_theo = dp * flow_theo / 600  # kW
        p_actual = p_theo - ploss / 1000  # kW
        nu_ges = p_actual / p_theo
        nu_hm = nu_ges / nu_vol

        # Display results
        print(f'\n--- PISTON ANALYSIS: {sim_name} ---')
        print(f'Speed       : {speed:.0f} rpm')
        print(f'HP / LP     : {HP:.2f} / {LP:.2f} bar (Δp = {dp:.2f} bar)')
        print(f'Beta        : {beta:.2f} / {max_beta:.2f} ({beta_norm * 100:.1f}%)')
        print(f'Flow Theo   : {flow_theo:.2f} l/min')
        print(f'Leakage     : {leakage:.2f} l/min')
        print(f'Net Flow    : {flow:.2f} l/min')
        print(f'Ploss Total : {ploss:.2f} W')
        print(f'Ploss Frict : {ploss_f:.2f} W')
        print(f'Ploss Leak  : {ploss_l:.2f} W')
        print(f'Power Theo  : {p_theo:.2f} kW')
        print(f'Power Actual: {p_actual:.2f} kW')
        print(f'Vol Eff     : {nu_vol * 100:.2f} %')
        print(f'Hm  Eff     : {nu_hm * 100:.2f} %')
        print(f'Ges Eff     : {nu_ges * 100:.2f} %')

        # Store results
        result = {
            'Folder': sim_name,
            'BaseFolder': folder_short_name,
            'PumpType': folder_label,
            'Displacement': displacement_volume,
            'Speed': speed,
            'Beta': beta,
            'MaxBeta': max_beta,
            'HP': HP,
            'LP': LP,
            'dK': dK,
            'FlowTheo': flow_theo,
            'Leakage': leakage,
            'NetFlow': flow,
            'PowerTheo': p_theo,
            'PowerActual': p_actual,
            'PlossTotal': ploss,
            'PlossFriction': ploss_f,
            'PlossLeakage': ploss_l,
            'VolEff': nu_vol * 100,
            'HmEff': nu_hm * 100,
            'GesEff': nu_ges * 100
        }

        results.append(result)

    return results


def create_comparison_plots(all_results):
    """Create comparison plots for all pump types"""
    if not all_results:
        print('\n⚠️ No valid simulation results processed for any folder.')
        return

    # Ensure output directory exists
    os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)

    df = pd.DataFrame(all_results)
    unique_pumps = df['PumpType'].unique()

    print(f'\n📊 Creating comparison plots for {len(unique_pumps)} pump types...')
    print(f'📁 Saving plots to: {PLOT_OUTPUT_FOLDER}')

    # Plot 1: Combined Speed vs Overall Efficiency
    try:
        fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size

        for i, pump_type in enumerate(unique_pumps):
            pump_data = df[df['PumpType'] == pump_type].copy()
            pump_data = pump_data.sort_values('Speed')

            ax.plot(pump_data['Speed'], pump_data['GesEff'], '-o',
                    linewidth=3.0, color=presentation_colors[i % len(presentation_colors)],
                    markersize=8, label=pump_type)  # Increased marker size

        ax.set_xlabel('Speed [RPM]', fontsize=18)
        ax.set_ylabel('Overall Efficiency [%]', fontsize=18)
        ax.set_title('Speed vs. Overall Efficiency in Piston - Bushing Interface\n'+"        Pressure= 350 bar , Displacement= 100%         ", fontsize=20)
        ax.set_ylim([80, 100])
        ax.grid(True, linestyle='--', alpha=0.3)

        # Enhanced legend formatting
        legend = ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
                           frameon=True, fancybox=True, shadow=False, framealpha=0.9, edgecolor='black')
        legend.get_frame().set_facecolor('white')

        plt.tight_layout()

        plot_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_Speed_vs_Efficiency_Presentation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
        eps_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_Speed_vs_Efficiency_Presentation.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight', transparent=True)

        print(f'📊 Combined Speed vs Efficiency plot saved')
        plt.close()

    except Exception as e:
        print(f'⚠️ Error creating combined efficiency plot: {e}')

    # Plot 2: Combined Speed vs Mechanical Power Loss
    try:
        fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size

        for i, pump_type in enumerate(unique_pumps):
            pump_data = df[df['PumpType'] == pump_type].copy()
            pump_data = pump_data.sort_values('Speed')

            ax.plot(pump_data['Speed'], pump_data['PlossFriction'], '-o',
                    linewidth=3.0, color=presentation_colors[i % len(presentation_colors)],
                    markersize=8, label=pump_type)  # Increased marker size

        ax.set_xlabel('Speed [RPM]', fontsize=18)
        ax.set_ylabel('Mechanical Power Loss [W]', fontsize=18)
        ax.set_title('Speed vs. Mechanical Power Loss \n'+"        Pressure= 350 bar , Displacement= 100%         ", fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Enhanced legend formatting
        legend = ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
                           frameon=True, fancybox=True, shadow=False, framealpha=0.9, edgecolor='black')
        legend.get_frame().set_facecolor('white')

        plt.tight_layout()

        plot_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_Speed_vs_MechanicalPowerLoss_Presentation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
        eps_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_Speed_vs_MechanicalPowerLoss_Presentation.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight', transparent=True)

        print(f'📊 Combined Speed vs Mechanical Power Loss plot saved')
        plt.close()

    except Exception as e:
        print(f'⚠️ Error creating combined mechanical power loss plot: {e}')

    # Plot 3: Combined Speed vs Volumetric Power Loss
    try:
        fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size

        for i, pump_type in enumerate(unique_pumps):
            pump_data = df[df['PumpType'] == pump_type].copy()
            pump_data = pump_data.sort_values('Speed')

            ax.plot(pump_data['Speed'], pump_data['PlossLeakage'], '-o',
                    linewidth=3.0, color=presentation_colors[i % len(presentation_colors)],
                    markersize=8, label=pump_type)  # Increased marker size

        ax.set_xlabel('Speed [RPM]', fontsize=18)
        ax.set_ylabel('Volumetric Power Loss [W]', fontsize=18)
        ax.set_title('Speed vs. Volumetric Power Loss \n'+"        Pressure= 350 bar , Displacement= 100%         ", fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Enhanced legend formatting
        legend = ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
                           frameon=True, fancybox=True, shadow=False, framealpha=0.9, edgecolor='black')
        legend.get_frame().set_facecolor('white')

        plt.tight_layout()

        plot_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_Speed_vs_VolumetricPowerLoss_Presentation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
        eps_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_Speed_vs_VolumetricPowerLoss_Presentation.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight', transparent=True)

        print(f'📊 Combined Speed vs Volumetric Power Loss plot saved')
        plt.close()

    except Exception as e:
        print(f'⚠️ Error creating combined volumetric power loss plot: {e}')

    # Plot 4: Compare Mech & Volumetric Loss at Matching Speeds Across Pump Types
    try:
        fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size

        # Find common speeds across all pumps
        common_speeds = None
        for pump_type in unique_pumps:
            pump_data = df[df['PumpType'] == pump_type]
            valid_data = pump_data.dropna(subset=['PlossFriction', 'PlossLeakage'])
            speeds = set(valid_data['Speed'].unique())

            if common_speeds is None:
                common_speeds = speeds
            else:
                common_speeds = common_speeds.intersection(speeds)

        if not common_speeds:
            print('⚠️ No matching speed values found across all pump types.')
        else:
            common_speeds = sorted(list(common_speeds))

            for i, pump_type in enumerate(unique_pumps):
                pump_data = df[df['PumpType'] == pump_type]
                valid_data = pump_data.dropna(subset=['PlossFriction', 'PlossLeakage'])
                matched_data = valid_data[valid_data['Speed'].isin(common_speeds)].copy()
                matched_data = matched_data.sort_values('Speed')

                color = presentation_colors[i % len(presentation_colors)]
                lighter_color = [c * 0.6 + 0.4 for c in color]

                # Mechanical loss
                ax.plot(matched_data['Speed'], matched_data['PlossFriction'], '-o',
                        linewidth=3.0, markersize=8, color=color,
                        label=f'{pump_type} - Mech')

                # Volumetric loss
                ax.plot(matched_data['Speed'], matched_data['PlossLeakage'], '--s',
                        linewidth=3.0, markersize=8, color=lighter_color,
                        label=f'{pump_type} - Vol')

            ax.set_xlabel('Speed [RPM]', fontsize=18)
            ax.set_ylabel('Power Loss [W]', fontsize=18)
            ax.set_title('Mech vs Volumetric Loss — Matching Speeds Across Pumps', fontsize=20)
            ax.grid(True, linestyle='--', alpha=0.3)

            # Enhanced legend formatting
            legend = ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
                               frameon=True, fancybox=True, shadow=False, framealpha=0.9, edgecolor='black')
            legend.get_frame().set_facecolor('white')

            plt.tight_layout()

            plot_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_Matched_Speed_Mech_Vol_Loss.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', transparent=True)
            eps_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_Matched_Speed_Mech_Vol_Loss.eps')
            plt.savefig(eps_path, format='eps', bbox_inches='tight', transparent=True)

            print(f'📊 Matched Speed Mech vs Vol Loss plot saved')
            plt.close()

    except Exception as e:
        print(f'⚠️ Error creating matched speed loss plot: {e}')


def main():
    """Main function to run the analysis"""
    # === USER INPUT ===
    base_folders = [
        r'Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run1_Test_V60N\Run_V60N',
        r'Z:\Studenten\Mit\Inline_Thesis-Simulation\V60N_inclined_pump\Run\Run2_Test_inclined_&_inclined_code-V60N\Run1'
    ]

    # Displacement volumes for each folder (cc)
    displacement_volumes = [110, 110]

    # Custom labels for the base folders
    custom_summary_labels = [
        "V60N Straight",
        "V60N Inclined 5°"
    ]

    # Initialize master result container
    all_results = []

    # === Process Each Base Folder ===
    for folder_idx, base_folder in enumerate(base_folders):
        V = displacement_volumes[folder_idx]
        folder_label = custom_summary_labels[folder_idx]

        # Clean the folder label
        folder_label = clean_folder_label(folder_label)

        if not folder_label or folder_label == 'Folder_Unknown':
            folder_label = f'Folder_{folder_idx + 1}'

        # Process folder
        results = process_folder(base_folder, V, folder_label)

        # Save results for this folder
        if results:
            df_results = pd.DataFrame(results)
            safe_label = clean_folder_label(folder_label)

            output_path = os.path.join(base_folder, f'{safe_label}_PistonEfficiencyResults.xlsx')

            try:
                df_results.to_excel(output_path, index=False)
                print(f'\n✅ Results for {folder_label} saved to:\n{output_path}')
            except Exception as e:
                print(f'⚠️ Error saving Excel file: {e}')
                csv_path = os.path.join(base_folder, f'{safe_label}_PistonEfficiencyResults.csv')
                try:
                    df_results.to_csv(csv_path, index=False)
                    print(f'✅ Results saved as CSV to: {csv_path}')
                except Exception as csv_err:
                    print(f'⚠️ Error saving CSV file: {csv_err}')

            all_results.extend(results)
        else:
            print(f'\n⚠️ No valid simulation results processed for {folder_label}.')

    # === Save Combined Results ===
    if all_results:
        # Save combined results to the specified output folder
        df_combined = pd.DataFrame(all_results)

        combined_output_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_PistonEfficiencyResults.xlsx')

        try:
            df_combined.to_excel(combined_output_path, index=False)
            print(f'\n✅ Combined results for all pump types saved to:\n{combined_output_path}')
        except Exception as e:
            print(f'⚠️ Error saving combined Excel file: {e}')
            csv_combined_path = os.path.join(PLOT_OUTPUT_FOLDER, 'All_Pumps_PistonEfficiencyResults.csv')
            try:
                df_combined.to_csv(csv_combined_path, index=False)
                print(f'✅ Combined results saved as CSV to: {csv_combined_path}')
            except Exception as csv_err:
                print(f'⚠️ Error saving combined CSV file: {csv_err}')

        # === Create Comparison Plots ===
        create_comparison_plots(all_results)

    else:
        print('\n⚠️ No valid simulation results processed for any folder.')

    print('\n🎉 Analysis complete with presentation-ready plots!')
    print('📋 Plot files generated with ENHANCED FONT SIZES for PowerPoint:')
    print('   • Base font size: 16pt (increased from 12pt)')
    print('   • Axis labels: 18pt (increased from 12pt)')
    print('   • Plot titles: 20pt (increased from 14pt)')
    print('   • Legend font: 16pt (increased from 12pt)')
    print('   • Tick labels: 16pt (added explicit sizing)')
    print('   • Thicker lines and larger markers for better visibility')
    print('   • PNG format (300 DPI) with transparent background')
    print('   • EPS format for high-quality vector graphics')
    print(f'   • All files saved to: {PLOT_OUTPUT_FOLDER}')


if __name__ == "__main__":
    main()