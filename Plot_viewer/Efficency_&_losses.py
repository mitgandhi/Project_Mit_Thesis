#!/usr/bin/env python3
"""
Batch script to analyze piston behavior & efficiency for multiple base folders
Converted from MATLAB to Python with enhanced plotting for thesis presentation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import re
import warnings
from typing import List, Dict, Tuple, Optional
import seaborn as sns

# Set plotting style for professional presentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for high-quality plots
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})


class PistonAnalyzer:
    def __init__(self):
        self.all_results = pd.DataFrame()

    def read_piston_data(self, file_path: str) -> Tuple[pd.DataFrame, bool]:
        """Read piston data from file"""
        try:
            # Try reading with tab delimiter first
            df = pd.read_csv(file_path, delimiter='\t', low_memory=False)

            # Clean column names
            df.columns = df.columns.str.replace('%', 'X_').str.strip()

            return df, True
        except Exception as e:
            print(f"⚠️ Error reading piston data from {file_path}: {e}")
            return pd.DataFrame(), False

    def read_geometry_data(self, file_path: str) -> float:
        """Read dK value from geometry file"""
        dK = np.nan
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('dK'):
                        match = re.search(r'dK\s+([\d\.]+)', line)
                        if match:
                            dK = float(match.group(1))
                            break
        except Exception as e:
            print(f"⚠️ Error reading geometry file {file_path}: {e}")

        return dK

    def read_operating_conditions(self, file_path: str) -> Dict[str, float]:
        """Read operating conditions from file"""
        conditions = {'speed': np.nan, 'beta': np.nan, 'maxBeta': np.nan,
                      'HP': np.nan, 'LP': np.nan}

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    for key in conditions.keys():
                        if line.startswith(key) and 'max' not in line.lower():
                            # Handle different formats
                            if key == 'maxBeta' and line.startswith('betamax'):
                                match = re.search(r'betamax\s+([\d\.]+)', line)
                            else:
                                match = re.search(rf'{key}\s+([\d\.]+)', line)

                            if match:
                                conditions[key] = float(match.group(1))
                            else:
                                # Try alternative parsing
                                try:
                                    val_str = line.split(key)[-1].strip()
                                    conditions[key] = float(val_str)
                                except:
                                    pass

                    # Special handling for betamax
                    if line.startswith('betamax'):
                        match = re.search(r'betamax\s+([\d\.]+)', line)
                        if match:
                            conditions['maxBeta'] = float(match.group(1))

        except Exception as e:
            print(f"⚠️ Error reading operating conditions from {file_path}: {e}")

        return conditions

    def calculate_efficiency(self, piston_data: pd.DataFrame, conditions: Dict[str, float],
                             displacement: float) -> Dict[str, float]:
        """Calculate efficiency metrics"""

        # Find column names (handle different naming conventions)
        rev_col = None
        leakage_col = None
        mech_loss_col = None
        vol_loss_col = None

        for col in piston_data.columns:
            if 'revolution' in col.lower():
                rev_col = col
            elif 'leakage' in col.lower() and 'total' in col.lower():
                leakage_col = col
            elif 'mechanical' in col.lower() and 'power' in col.lower() and 'loss' in col.lower():
                mech_loss_col = col
            elif 'volumetric' in col.lower() and 'power' in col.lower() and 'loss' in col.lower():
                vol_loss_col = col

        if rev_col is None:
            raise ValueError("Revolution column not found")

        # Get last revolution data
        revs = piston_data[rev_col]
        max_rev = int(np.floor(revs.max()))

        if max_rev < 1:
            raise ValueError("Not enough data for full revolution")

        # Find indices for last revolution
        mask = (revs >= max_rev - 1) & (revs <= max_rev)
        last_rev_data = piston_data[mask]

        # Calculate basic parameters
        speed = conditions['speed']
        beta = conditions['beta']
        max_beta = conditions['maxBeta']
        HP = conditions['HP']
        LP = conditions['LP']
        dp = HP - LP

        beta_norm = beta / max_beta
        flow_theo = speed * displacement * beta_norm / 1000  # l/min

        # Get leakage
        if leakage_col is not None:
            leakage = last_rev_data[leakage_col].mean() * 60000  # l/min
        else:
            leakage = 0
            print("⚠️ Leakage column not found - assuming zero")

        flow = flow_theo - leakage
        nu_vol = flow / flow_theo if flow_theo > 0 else 0

        # Get power losses
        if mech_loss_col is not None:
            ploss_f = last_rev_data[mech_loss_col].mean()
        else:
            ploss_f = 0
            print("⚠️ Mechanical loss column not found - assuming zero")

        if vol_loss_col is not None:
            ploss_l = last_rev_data[vol_loss_col].mean()
        else:
            ploss_l = leakage * dp / 600 * 1000  # W
            print("⚠️ Volumetric loss column not found - estimating from leakage")

        ploss_total = ploss_f + ploss_l

        # Calculate efficiencies
        P_theo = dp * flow_theo / 600  # kW
        P_actual = P_theo - ploss_total / 1000  # kW
        nu_ges = P_actual / P_theo if P_theo > 0 else 0
        nu_hm = nu_ges / nu_vol if nu_vol > 0 else 0

        return {
            'flow_theo': flow_theo,
            'leakage': leakage,
            'net_flow': flow,
            'P_theo': P_theo,
            'P_actual': P_actual,
            'ploss_total': ploss_total,
            'ploss_friction': ploss_f,
            'ploss_leakage': ploss_l,
            'nu_vol': nu_vol * 100,
            'nu_hm': nu_hm * 100,
            'nu_ges': nu_ges * 100,
            'dp': dp
        }

    def create_professional_plot(self, title: str, xlabel: str, ylabel: str,
                                 figsize: Tuple[int, int] = (12, 8)) -> Tuple:
        """Create a professional plot for thesis presentation"""
        fig, ax = plt.subplots(figsize=figsize)

        # Customize the plot appearance
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=16, fontweight='semibold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='semibold')

        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Spine styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('gray')

        return fig, ax

    def plot_speed_vs_efficiency(self, results: pd.DataFrame, pump_type: str,
                                 output_dir: str, individual: bool = True):
        """Create speed vs efficiency plot for thesis presentation"""
        if results.empty:
            return

        # Sort by speed
        results_sorted = results.sort_values('Speed')

        fig, ax = self.create_professional_plot(
            f'Speed vs. Overall Efficiency - {pump_type}',
            'Speed [RPM]',
            'Overall Efficiency [%]'
        )

        # Main plot with enhanced styling
        colors = sns.color_palette("husl", 1)
        ax.scatter(results_sorted['Speed'], results_sorted['GesEff'],
                   s=120, alpha=0.8, color=colors[0], edgecolors='black',
                   linewidth=1.5, zorder=5)
        ax.plot(results_sorted['Speed'], results_sorted['GesEff'],
                color=colors[0], linewidth=3, alpha=0.8, zorder=4)

        # Customize axes
        ax.set_ylim(60, 100)
        speed_range = results['Speed'].max() - results['Speed'].min()
        ax.set_xlim(results['Speed'].min() - speed_range * 0.05,
                    results['Speed'].max() + speed_range * 0.05)

        # Add value annotations on key points
        max_eff_idx = results_sorted['GesEff'].idxmax()
        max_eff_point = results_sorted.loc[max_eff_idx]
        ax.annotate(f'{max_eff_point["GesEff"]:.1f}%',
                    (max_eff_point['Speed'], max_eff_point['GesEff']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=12, fontweight='bold')

        plt.tight_layout()

        if individual:
            filename = f'{pump_type.replace(" ", "_")}_Speed_vs_Efficiency.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f'📊 Speed vs Efficiency plot saved to: {filepath}')

        return fig

    def plot_power_losses(self, results: pd.DataFrame, pump_type: str,
                          output_dir: str, loss_type: str = 'mechanical'):
        """Create power loss plots for thesis presentation"""
        if results.empty:
            return

        results_sorted = results.sort_values('Speed')

        if loss_type == 'mechanical':
            y_data = results_sorted['PlossFriction']
            ylabel = 'Mechanical Power Loss [W]'
            title = f'Speed vs. Mechanical Power Loss - {pump_type}'
            color = 'red'
        else:
            y_data = results_sorted['PlossLeakage']
            ylabel = 'Volumetric Power Loss [W]'
            title = f'Speed vs. Volumetric Power Loss - {pump_type}'
            color = 'blue'

        fig, ax = self.create_professional_plot(title, 'Speed [RPM]', ylabel)

        # Plot with enhanced styling
        ax.scatter(results_sorted['Speed'], y_data,
                   s=120, alpha=0.8, color=color, edgecolors='black',
                   linewidth=1.5, zorder=5)
        ax.plot(results_sorted['Speed'], y_data,
                color=color, linewidth=3, alpha=0.8, zorder=4)

        # Customize axes
        speed_range = results['Speed'].max() - results['Speed'].min()
        ax.set_xlim(results['Speed'].min() - speed_range * 0.05,
                    results['Speed'].max() + speed_range * 0.05)

        y_range = y_data.max() - y_data.min()
        ax.set_ylim(max(0, y_data.min() - y_range * 0.1),
                    y_data.max() + y_range * 0.1)

        plt.tight_layout()

        filename = f'{pump_type.replace(" ", "_")}_Speed_vs_{loss_type.title()}PowerLoss.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'📊 Speed vs {loss_type.title()} Power Loss plot saved to: {filepath}')

        return fig

    def plot_combined_results(self, all_results: pd.DataFrame, output_dir: str,
                              plot_type: str = 'efficiency'):
        """Create combined plots for all pump types"""
        if all_results.empty:
            return

        unique_pumps = all_results['PumpType'].unique()
        colors = sns.color_palette("husl", len(unique_pumps))

        if plot_type == 'efficiency':
            title = 'Speed vs. Overall Efficiency - All Pump Types'
            ylabel = 'Overall Efficiency [%]'
            y_col = 'GesEff'
            filename = 'All_Pumps_Speed_vs_Efficiency.png'
        elif plot_type == 'mechanical_loss':
            title = 'Speed vs. Mechanical Power Loss - All Pump Types'
            ylabel = 'Mechanical Power Loss [W]'
            y_col = 'PlossFriction'
            filename = 'All_Pumps_Speed_vs_MechanicalPowerLoss.png'
        else:  # volumetric_loss
            title = 'Speed vs. Volumetric Power Loss - All Pump Types'
            ylabel = 'Volumetric Power Loss [W]'
            y_col = 'PlossLeakage'
            filename = 'All_Pumps_Speed_vs_VolumetricPowerLoss.png'

        fig, ax = self.create_professional_plot(title, 'Speed [RPM]', ylabel)

        for i, pump_type in enumerate(unique_pumps):
            pump_data = all_results[all_results['PumpType'] == pump_type].sort_values('Speed')

            ax.plot(pump_data['Speed'], pump_data[y_col],
                    '-o', linewidth=3, markersize=10, color=colors[i],
                    label=pump_type, alpha=0.8)

        # Customize legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True,
                  framealpha=0.9, fontsize=14)

        if plot_type == 'efficiency':
            ax.set_ylim(60, 100)

        plt.tight_layout()

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'📊 Combined {plot_type} plot saved to: {filepath}')

        return fig

    def analyze_folder(self, base_folder: str, displacement: float,
                       pump_type: str) -> pd.DataFrame:
        """Analyze all simulations in a base folder"""
        print(f'\n{"=" * 70}')
        print(f'PROCESSING: {pump_type}')
        print(f'Folder: {base_folder}')
        print(f'Displacement Volume: {displacement} cc')
        print(f'{"=" * 70}\n')

        results = []

        # Get all subfolders
        base_path = Path(base_folder)
        if not base_path.exists():
            print(f'⚠️ Base folder does not exist: {base_folder}')
            return pd.DataFrame()

        subfolders = [d for d in base_path.iterdir() if d.is_dir()
                      and not d.name.startswith('.')
                      and not d.name.startswith('Thumbs')
                      and 'Results' not in d.name]

        for subfolder in subfolders:
            sim_name = subfolder.name
            piston_file = subfolder / 'output' / 'piston' / 'piston.txt'

            if not piston_file.exists():
                print(f'Skipping {sim_name} — piston.txt not found.')
                continue

            print(f'Processing: {sim_name}')

            # Read geometry data
            geom_file = subfolder / 'input' / 'geometry.txt'
            dK = self.read_geometry_data(str(geom_file)) if geom_file.exists() else np.nan

            # Read operating conditions
            op_file = subfolder / 'input' / 'operatingconditions.txt'
            if not op_file.exists():
                print(f'⚠️ operatingconditions.txt not found for {sim_name} — skipping.')
                continue

            conditions = self.read_operating_conditions(str(op_file))

            # Check if all required conditions are available
            required_conditions = ['speed', 'beta', 'maxBeta', 'HP', 'LP']
            if any(np.isnan(conditions[key]) for key in required_conditions):
                print(f'⚠️ Incomplete operating conditions in {sim_name} — skipping.')
                continue

            # Read piston data
            piston_data, success = self.read_piston_data(str(piston_file))
            if not success or piston_data.empty:
                continue

            try:
                # Calculate efficiency metrics
                metrics = self.calculate_efficiency(piston_data, conditions, displacement)

                # Display results
                print(f'\n--- PISTON ANALYSIS: {sim_name} ---')
                print(f'Speed       : {conditions["speed"]:.0f} rpm')
                print(
                    f'HP / LP     : {conditions["HP"]:.2f} / {conditions["LP"]:.2f} bar (Δp = {metrics["dp"]:.2f} bar)')
                print(
                    f'Beta        : {conditions["beta"]:.2f} / {conditions["maxBeta"]:.2f} ({conditions["beta"] / conditions["maxBeta"] * 100:.1f}%)')
                print(f'Flow Theo   : {metrics["flow_theo"]:.2f} l/min')
                print(f'Leakage     : {metrics["leakage"]:.2f} l/min')
                print(f'Net Flow    : {metrics["net_flow"]:.2f} l/min')
                print(f'Ploss Total : {metrics["ploss_total"]:.2f} W')
                print(f'Ploss Frict : {metrics["ploss_friction"]:.2f} W')
                print(f'Ploss Leak  : {metrics["ploss_leakage"]:.2f} W')
                print(f'Power Theo  : {metrics["P_theo"]:.2f} kW')
                print(f'Power Actual: {metrics["P_actual"]:.2f} kW')
                print(f'Vol Eff     : {metrics["nu_vol"]:.2f} %')
                print(f'Hm  Eff     : {metrics["nu_hm"]:.2f} %')
                print(f'Ges Eff     : {metrics["nu_ges"]:.2f} %\n')

                # Store results
                result = {
                    'Folder': sim_name,
                    'BaseFolder': base_path.name,
                    'PumpType': pump_type,
                    'Displacement': displacement,
                    'Speed': conditions['speed'],
                    'Beta': conditions['beta'],
                    'MaxBeta': conditions['maxBeta'],
                    'HP': conditions['HP'],
                    'LP': conditions['LP'],
                    'dK': dK,
                    'FlowTheo': metrics['flow_theo'],
                    'Leakage': metrics['leakage'],
                    'NetFlow': metrics['net_flow'],
                    'PowerTheo': metrics['P_theo'],
                    'PowerActual': metrics['P_actual'],
                    'PlossTotal': metrics['ploss_total'],
                    'PlossFriction': metrics['ploss_friction'],
                    'PlossLeakage': metrics['ploss_leakage'],
                    'VolEff': metrics['nu_vol'],
                    'HmEff': metrics['nu_hm'],
                    'GesEff': metrics['nu_ges']
                }

                results.append(result)

            except Exception as e:
                print(f'⚠️ Error processing {sim_name}: {e}')
                continue

        if results:
            df_results = pd.DataFrame(results)

            # Save individual results
            safe_pump_type = re.sub(r'[<>:"/\\|?*]', '_', pump_type)
            excel_path = base_path / f'{safe_pump_type}_PistonEfficiencyResults.xlsx'

            try:
                df_results.to_excel(str(excel_path), index=False)
                print(f'✅ Results saved to: {excel_path}')
            except Exception as e:
                print(f'⚠️ Error saving Excel file: {e}')
                csv_path = base_path / f'{safe_pump_type}_PistonEfficiencyResults.csv'
                df_results.to_csv(str(csv_path), index=False)
                print(f'✅ Results saved as CSV to: {csv_path}')

            # Create individual plots
            self.plot_speed_vs_efficiency(df_results, pump_type, str(base_path))
            self.plot_power_losses(df_results, pump_type, str(base_path), 'mechanical')
            self.plot_power_losses(df_results, pump_type, str(base_path), 'volumetric')

            return df_results
        else:
            print(f'⚠️ No valid simulation results processed for {pump_type}.')
            return pd.DataFrame()

    def run_analysis(self, base_folders: List[str], displacement_volumes: List[float],
                     custom_labels: List[str]):
        """Run the complete analysis"""
        print("🚀 Starting Piston Analysis")
        print(f"Processing {len(base_folders)} base folders...")

        all_results = []

        for i, (folder, displacement, label) in enumerate(zip(base_folders, displacement_volumes, custom_labels)):
            folder_results = self.analyze_folder(folder, displacement, label)
            if not folder_results.empty:
                all_results.append(folder_results)

        if all_results:
            # Combine all results
            combined_results = pd.concat(all_results, ignore_index=True)
            self.all_results = combined_results

            # Save combined results
            parent_dir = Path(base_folders[0]).parent
            combined_excel_path = parent_dir / 'All_Pumps_PistonEfficiencyResults.xlsx'

            try:
                combined_results.to_excel(str(combined_excel_path), index=False)
                print(f'✅ Combined results saved to: {combined_excel_path}')
            except Exception as e:
                print(f'⚠️ Error saving combined Excel: {e}')
                combined_csv_path = parent_dir / 'All_Pumps_PistonEfficiencyResults.csv'
                combined_results.to_csv(str(combined_csv_path), index=False)
                print(f'✅ Combined results saved as CSV to: {combined_csv_path}')

            # Create combined plots
            self.plot_combined_results(combined_results, str(parent_dir), 'efficiency')
            self.plot_combined_results(combined_results, str(parent_dir), 'mechanical_loss')
            self.plot_combined_results(combined_results, str(parent_dir), 'volumetric_loss')

            print(f'\n🎉 Analysis complete for all {len(base_folders)} base folders.')
            return combined_results
        else:
            print('\n⚠️ No valid simulation results processed for any folder.')
            return pd.DataFrame()


def main():
    """Main execution function"""

    # === USER INPUT ===
    base_folders = [
        'Z:\\Studenten\\Mit\\Inline_Thesis-Simulation\\V60N_inclined_pump\\Run\\Run1_Test_V60N\\Run_V60N'
    ]

    # Displacement volumes for each folder (cc)
    displacement_volumes = [110]

    # Custom labels for the base folders
    custom_labels = ["V60N"]

    # Validate input
    if len(base_folders) != len(displacement_volumes) or len(base_folders) != len(custom_labels):
        raise ValueError("Number of base folders, displacement volumes, and labels must match")

    # Initialize analyzer
    analyzer = PistonAnalyzer()

    # Run analysis
    results = analyzer.run_analysis(base_folders, displacement_volumes, custom_labels)

    # Show summary
    if not results.empty:
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total simulations processed: {len(results)}")
        print(f"Pump types analyzed: {results['PumpType'].nunique()}")
        print(f"Speed range: {results['Speed'].min():.0f} - {results['Speed'].max():.0f} RPM")
        print(f"Efficiency range: {results['GesEff'].min():.1f} - {results['GesEff'].max():.1f} %")
        print("=" * 50)


if __name__ == "__main__":
    main()