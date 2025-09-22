# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re
# from typing import Dict
# import numpy as np
# from scipy.stats import pearsonr
# import os
#
# # Import all functions from T1
# import sys
#
# sys.path.append('.')  # Add current directory to path
# from T1 import (
#     load_matlab_txt, round2, parse_geometry, parse_operating_conditions,
#     create_cumulative_pressure_map, collect_simulation_data,
#     create_comparison_plots, create_overall_summary, run_contact_pressure_analysis
# )
#
# # --- Data Parsing Functions (from original T2) ---
#
# PARAM_PATTERN = re.compile(
#     r"dK(?P<dK>[-\d\.]+)_dZ(?P<dZ>[-\d\.]+)_LKG(?P<LKG>[-\d\.]+)_lF(?P<lF>[-\d\.]+)_zeta(?P<zeta>[-\d\.]+)"
# )
#
#
# def parse_parameters(folder_name: str) -> Dict[str, float]:
#     match = PARAM_PATTERN.search(folder_name)
#     if not match:
#         return {}
#     try:
#         params = {k: float(v) for k, v in match.groupdict().items()}
#         params["zeta"] = float(int(params["zeta"]))
#         return params
#     except ValueError:
#         return {}
#
#
# def parse_loss_file(sim_path: str) -> Dict[str, float]:
#     piston_path = Path(sim_path) / "output" / "piston" / "piston.txt"
#     if not piston_path.exists():
#         return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
#     try:
#         df_loss = pd.read_csv(piston_path, delimiter="\t")
#         df_loss = df_loss[df_loss["revolution"] <= 6.0]
#         if df_loss.empty:
#             return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
#         mech = abs(df_loss["Total_Mechanical_Power_Loss"].max())
#         vol = abs(df_loss["Total_Volumetric_Power_Loss"].max())
#         if pd.isna(mech) or pd.isna(vol):
#             return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
#         return {"mechanical": mech, "volumetric": vol, "total": mech + vol, "valid": True}
#     except Exception:
#         return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
#
#
# def load_results(folder_path: str, opt_type: str) -> list:
#     base_folder = Path(folder_path)
#     results = []
#     if not base_folder.exists():
#         print(f"Base folder not found: {base_folder}")
#         return []
#     subfolders = [f for f in base_folder.iterdir() if f.is_dir()]
#     for folder in subfolders:
#         folder_name = folder.name
#         if opt_type == "NSGA-III":
#             if folder_name.startswith("Generation_G"):
#                 try:
#                     iter_num = int(folder_name.replace("Generation_G", ""))
#                     iter_type = "generation"
#                 except ValueError:
#                     continue
#             elif folder_name == "Initial_Sampling":
#                 iter_num = 0
#                 iter_type = "generation"
#             else:
#                 continue
#         elif opt_type == "BO":
#             if folder_name.startswith("Iteration_I"):
#                 try:
#                     iter_num = int(folder_name.replace("Iteration_I", ""))
#                     iter_type = "iteration"
#                 except ValueError:
#                     continue
#             elif folder_name == "Initial_Sampling":
#                 iter_num = 0
#                 iter_type = "iteration"
#             else:
#                 continue
#         else:
#             print(f"Unknown optimization type: {opt_type}")
#             continue
#
#         sim_folders = [f for f in folder.iterdir() if f.is_dir()]
#         for sim in sim_folders:
#             if not PARAM_PATTERN.search(sim.name):
#                 continue
#             params = parse_parameters(sim.name)
#             if not params:
#                 continue
#             losses = parse_loss_file(str(sim))
#             record = {**params, **losses}
#             record[iter_type] = iter_num
#             record["folder_name"] = str(sim)
#             record["optimizer"] = opt_type
#             if record.get("valid") and record["total"] < 1e6:
#                 results.append(record)
#     return results
#
#
# def find_pareto_front(df: pd.DataFrame, objective_cols: list, minimize: bool = True) -> pd.DataFrame:
#     """
#     Identifies the Pareto front from a DataFrame of solutions.
#     """
#     if df.empty or not all(col in df.columns for col in objective_cols):
#         return pd.DataFrame()
#
#     df_sorted = df.sort_values(by=objective_cols[0], ascending=minimize).reset_index(drop=True)
#
#     pareto_indices = []
#     for i in range(len(df_sorted)):
#         is_pareto = True
#         for j in range(len(df_sorted)):
#             if i == j:
#                 continue
#
#             dominates_all_objectives = True
#             strictly_better_in_one = False
#
#             for obj in objective_cols:
#                 if minimize:
#                     if df_sorted.loc[j, obj] > df_sorted.loc[i, obj]:
#                         dominates_all_objectives = False
#                         break
#                     if df_sorted.loc[j, obj] < df_sorted.loc[i, obj]:
#                         strictly_better_in_one = True
#                 else:
#                     if df_sorted.loc[j, obj] < df_sorted.loc[i, obj]:
#                         dominates_all_objectives = False
#                         break
#                     if df_sorted.loc[j, obj] > df_sorted.loc[i, obj]:
#                         strictly_better_in_one = True
#
#             if dominates_all_objectives and strictly_better_in_one:
#                 is_pareto = False
#                 break
#
#         if is_pareto:
#             pareto_indices.append(i)
#
#     return df_sorted.loc[pareto_indices].drop_duplicates().reset_index(drop=True)
#
#
# def check_required_files(folder_path):
#     """
#     Check if all required files exist for contact pressure analysis.
#     Returns both missing files and available alternatives.
#     """
#     # Primary files needed for full T1 analysis
#     primary_files = [
#         os.path.join(folder_path, 'input', 'geometry.txt'),
#         os.path.join(folder_path, 'input', 'operatingconditions.txt'),
#         os.path.join(folder_path, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt'),
#         os.path.join(folder_path, 'output', 'piston', 'piston.txt')
#     ]
#
#     # Alternative files that might exist
#     alternative_files = [
#         os.path.join(folder_path, 'output', 'piston', 'piston.txt'),  # Main results file
#         os.path.join(folder_path, 'output', 'piston.txt'),  # Alternative location
#         os.path.join(folder_path, 'output', 'results.txt'),  # Generic results
#     ]
#
#     missing_primary = []
#     available_alternatives = []
#
#     for file_path in primary_files:
#         if not os.path.exists(file_path):
#             missing_primary.append(file_path)
#
#     for file_path in alternative_files:
#         if os.path.exists(file_path):
#             available_alternatives.append(file_path)
#
#     return missing_primary, available_alternatives
#
#
# def extract_contact_pressure_from_piston_file(piston_file_path):
#     """
#     Extract contact pressure information from piston.txt file if available.
#     This is a fallback when the matlab contact pressure files don't exist.
#     """
#     try:
#         import pandas as pd
#         df = pd.read_csv(piston_file_path, delimiter='\t')
#
#         print(f"    📋 File columns: {list(df.columns)}")
#
#         # Look for contact pressure related columns with more specific patterns
#         potential_cp_cols = []
#         for col in df.columns:
#             col_lower = col.lower()
#             if any(keyword in col_lower for keyword in ['contact', 'pressure', 'force', 'load']):
#                 if not any(exclude in col_lower for exclude in ['loss', 'power', 'volume']):
#                     potential_cp_cols.append(col)
#
#         print(f"    🔍 Potential contact pressure columns: {potential_cp_cols}")
#
#         if potential_cp_cols:
#             # Use the revolution data (similar to T2 analysis)
#             df_filtered = df[df['revolution'] <= 6.0] if 'revolution' in df.columns else df
#             print(f"    📊 Data rows: {len(df)} total, {len(df_filtered)} filtered")
#
#             contact_pressure_data = {}
#             for col in potential_cp_cols:
#                 if df_filtered[col].notna().any():
#                     col_mean = float(df_filtered[col].mean())
#                     col_max = float(df_filtered[col].max())
#                     col_min = float(df_filtered[col].min())
#
#                     contact_pressure_data[col] = {
#                         'mean': col_mean,
#                         'max': col_max,
#                         'min': col_min
#                     }
#                     print(f"    📈 {col}: mean={col_mean:.2e}, max={col_max:.2e}, min={col_min:.2e}")
#
#             return contact_pressure_data
#         else:
#             print(f"    ❌ No contact pressure related columns found")
#             return None
#
#     except Exception as e:
#         print(f"    ❌ Error reading piston file: {e}")
#         return None
#
#
# def check_file_structure(folder_path):
#     """
#     Check what files actually exist and their structure.
#     """
#     print(f"  🔍 Checking file structure for: {os.path.basename(folder_path)}")
#
#     # Check all possible locations
#     file_locations = [
#         ('input/geometry.txt', os.path.join(folder_path, 'input', 'geometry.txt')),
#         ('input/operatingconditions.txt', os.path.join(folder_path, 'input', 'operatingconditions.txt')),
#         ('output/piston/matlab/Piston_Contact_Pressure.txt',
#          os.path.join(folder_path, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt')),
#         ('output/piston/piston.txt', os.path.join(folder_path, 'output', 'piston', 'piston.txt')),
#         ('output/piston.txt', os.path.join(folder_path, 'output', 'piston.txt')),
#     ]
#
#     for name, path in file_locations:
#         exists = os.path.exists(path)
#         size = os.path.getsize(path) if exists else 0
#         print(
#             f"    {'✅' if exists else '❌'} {name}: {'exists' if exists else 'missing'} {f'({size} bytes)' if exists else ''}")
#
#     # Check if matlab directory exists at all
#     matlab_dir = os.path.join(folder_path, 'output', 'piston', 'matlab')
#     if os.path.exists(matlab_dir):
#         matlab_files = os.listdir(matlab_dir)
#         print(f"    📁 Matlab directory contents: {matlab_files}")
#     else:
#         print(f"    ❌ Matlab directory doesn't exist: {matlab_dir}")
#
#
# def run_contact_pressure_for_best_results(best_folders, output_dir):
#     """
#     Run contact pressure analysis for the best simulation folders using available data.
#     Falls back to piston.txt analysis when full T1 analysis isn't possible.
#     """
#     print("\n" + "=" * 60)
#     print("RUNNING CONTACT PRESSURE ANALYSIS FOR BEST RESULTS")
#     print("=" * 60)
#
#     contact_pressure_results = []
#     skipped_count = 0
#     fallback_count = 0
#
#     # First, let's check a few folders to understand the structure
#     print("\n🔍 DIAGNOSTIC: Checking file structure for first few folders...")
#     for i, folder_info in enumerate(best_folders[:3]):  # Check first 3 folders
#         folder_path = folder_info['folder_name']
#         check_file_structure(folder_path)
#         print()
#
#     for i, folder_info in enumerate(best_folders):
#         folder_path = folder_info['folder_name']
#         optimizer = folder_info['optimizer']
#
#         print(f"\n[{i + 1}/{len(best_folders)}] Processing {optimizer}: {os.path.basename(folder_path)}")
#
#         # Check if required files exist
#         missing_files, available_alternatives = check_required_files(folder_path)
#
#         # Try full T1 analysis first
#         can_do_full_analysis = len(missing_files) == 0
#
#         if can_do_full_analysis:
#             try:
#                 # Extract parameters from folder for T1 analysis
#                 geom_file = os.path.join(folder_path, 'input', 'geometry.txt')
#                 op_file = os.path.join(folder_path, 'input', 'operatingconditions.txt')
#
#                 lF_val = parse_geometry(geom_file)
#                 speed_val, hp_val = parse_operating_conditions(op_file)
#
#                 if lF_val is None or speed_val is None or hp_val is None:
#                     print(f"  ❌ Could not parse geometry or operating conditions")
#                     skipped_count += 1
#                     continue
#
#                 print(f"  📊 Parameters: lF={lF_val:.1f}mm, speed={speed_val:.0f}rpm, ΔP={hp_val:.1f}")
#                 print(f"  🔄 Running full T1 contact pressure analysis...")
#
#                 # Run contact pressure analysis using T1 function
#                 result = create_cumulative_pressure_map(
#                     filepath=folder_path,
#                     m=50,  # Standard value from T1
#                     lF=lF_val,
#                     n=speed_val,
#                     deltap=hp_val,
#                     plots=360,  # Standard value from T1
#                     degstep=1,  # Standard value from T1
#                     ignore=0,  # Standard value from T1
#                     offset=0,  # Standard value from T1
#                     minplot=0,  # Auto-calculate
#                     maxplot=0  # Auto-calculate
#                 )
#
#                 if result is not None:
#                     # Extract mean cumulative contact pressure
#                     mean_cumulative_pressure = np.mean(result['cumulative'])
#                     max_cumulative_pressure = np.max(result['cumulative'])
#
#                     contact_pressure_results.append({
#                         'folder_name': folder_path,
#                         'optimizer': optimizer,
#                         'analysis_type': 'Full T1 Analysis',
#                         'max_cumulative_pressure': max_cumulative_pressure,
#                         'mean_cumulative_pressure': mean_cumulative_pressure,
#                         'mechanical': folder_info['mechanical'],
#                         'volumetric': folder_info['volumetric'],
#                         'total': folder_info['total'],
#                         'dK': folder_info['dK'],
#                         'dZ': folder_info['dZ'],
#                         'LKG': folder_info['LKG'],
#                         'lF': folder_info['lF'],
#                         'zeta': folder_info['zeta'],
#                         'result': result
#                     })
#                     print(
#                         f"  ✅ Success - Mean: {mean_cumulative_pressure:.2e} Pa, Max: {max_cumulative_pressure:.2e} Pa")
#                     continue
#                 else:
#                     print(f"  ⚠️ T1 analysis returned None, trying fallback method...")
#
#             except Exception as e:
#                 print(f"  ⚠️ T1 analysis failed: {str(e)[:80]}..., trying fallback method...")
#
#         # Fallback: Try to extract contact pressure from piston.txt
#         piston_files_to_try = [
#             os.path.join(folder_path, 'output', 'piston', 'piston.txt'),
#             os.path.join(folder_path, 'output', 'piston.txt')
#         ]
#
#         fallback_success = False
#         for piston_file in piston_files_to_try:
#             if os.path.exists(piston_file):
#                 print(f"  🔄 Attempting fallback analysis from: {os.path.relpath(piston_file, folder_path)}")
#
#                 contact_data = extract_contact_pressure_from_piston_file(piston_file)
#                 if contact_data:
#                     # Choose the best contact pressure column (prefer "contact" in name)
#                     best_col = None
#                     for col in contact_data.keys():
#                         if 'contact' in col.lower():
#                             best_col = col
#                             break
#                     if not best_col:
#                         best_col = list(contact_data.keys())[0]  # Use first available
#
#                     cp_stats = contact_data[best_col]
#
#                     contact_pressure_results.append({
#                         'folder_name': folder_path,
#                         'optimizer': optimizer,
#                         'analysis_type': f'Fallback ({best_col})',
#                         'max_cumulative_pressure': cp_stats['max'],
#                         'mean_cumulative_pressure': cp_stats['mean'],
#                         'mechanical': folder_info['mechanical'],
#                         'volumetric': folder_info['volumetric'],
#                         'total': folder_info['total'],
#                         'dK': folder_info['dK'],
#                         'dZ': folder_info['dZ'],
#                         'LKG': folder_info['LKG'],
#                         'lF': folder_info['lF'],
#                         'zeta': folder_info['zeta'],
#                         'result': None  # No full T1 result available
#                     })
#                     print(f"  ✅ Fallback success - Mean: {cp_stats['mean']:.2e} Pa, Max: {cp_stats['max']:.2e} Pa")
#                     fallback_count += 1
#                     fallback_success = True
#                     break
#                 else:
#                     print(f"    ❌ No contact pressure columns found in {os.path.basename(piston_file)}")
#
#         if not fallback_success:
#             print(f"  ❌ No usable contact pressure data found")
#             if missing_files:
#                 print(f"    Missing files: {len(missing_files)} files")
#                 for missing_file in missing_files[:2]:  # Show first 2 missing files
#                     print(f"    - {os.path.relpath(missing_file, folder_path)}")
#             skipped_count += 1
#
#     print(f"\n✅ Contact pressure analysis completed:")
#     print(f"   - Full T1 analysis: {len(contact_pressure_results) - fallback_count} simulations")
#     print(f"   - Fallback analysis: {fallback_count} simulations")
#     print(f"   - Total successful: {len(contact_pressure_results)} simulations")
#     print(f"   - Skipped: {skipped_count} simulations")
#     print(f"   - Total processed: {len(best_folders)} simulations")
#
#     return contact_pressure_results
#
#
# # def create_contact_pressure_plots(contact_results, output_dir):
# #     """
# #     Create plots showing contact pressure analysis results.
# #     """
# #     if not contact_results:
# #         print("❌ No contact pressure results to plot - all simulations were skipped due to missing files")
# #         print("\n💡 To fix this issue:")
# #         print("   1. Ensure the simulation data files exist in the expected locations")
# #         print("   2. Check that simulations have completed successfully")
# #         print("   3. Verify the file paths in the folder structure")
# #         return
# #
# #     print(f"\n📊 Creating contact pressure plots for {len(contact_results)} successful simulations...")
# #
# #     df_contact = pd.DataFrame(contact_results)
# #
# #     # 1. Mean cumulative contact pressure comparison
# #     plt.figure(figsize=(12, 6))
# #
# #     # Separate by optimizer
# #     bo_data = df_contact[df_contact['optimizer'] == 'BO']
# #     nsga3_data = df_contact[df_contact['optimizer'] == 'NSGA-III']
# #
# #     # Create positions for bars
# #     x_positions = []
# #     labels = []
# #     colors = []
# #     pressures = []
# #
# #     # Add BO data
# #     for i, (_, row) in enumerate(bo_data.iterrows()):
# #         x_positions.append(i)
# #         labels.append(f"BO-{i + 1}")
# #         colors.append('blue')
# #         pressures.append(row['max_cumulative_pressure'])
# #
# #     # Add NSGA-III data
# #     nsga_start = len(bo_data)
# #     for i, (_, row) in enumerate(nsga3_data.iterrows()):
# #         x_positions.append(nsga_start + i)
# #         labels.append(f"NSGA-{i + 1}")
# #         colors.append('orange')
# #         pressures.append(row['max_cumulative_pressure'])
# #
# #     if pressures:  # Only create plot if we have data
# #         plt.bar(x_positions, pressures, color=colors, alpha=0.7, width=0.8)
# #         plt.xlabel('Simulation', fontsize=16)
# #         plt.ylabel('Max Cumulative Contact Pressure [Pa]', fontsize=16)
# #         plt.title(
# #             f'Max Cumulative Contact Pressure for Best Optimization Results\n({len(bo_data)} BO + {len(nsga3_data)} NSGA-III)',
# #             fontsize=14)
# #         plt.yscale('log')
# #         plt.xticks(x_positions, labels, rotation=45)
# #
# #         # Add legend
# #         bo_patch = plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label=f'BO ({len(bo_data)})')
# #         nsga3_patch = plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.7, label=f'NSGA-III ({len(nsga3_data)})')
# #         plt.legend(handles=[bo_patch, nsga3_patch], fontsize=14)
# #
# #         plt.grid(True, alpha=0.3)
# #         plt.xticks(fontsize=12)
# #         plt.yticks(fontsize=14)
# #         plt.tight_layout()
# #         plt.savefig(output_dir / "contact_pressure_comparison.png", dpi=300)
# #         plt.close()
# #
# #     # 2. Contact pressure vs volumetric loss (similar to pareto plot style)
# #     plt.figure(figsize=(8, 6))
# #
# #     if not bo_data.empty:
# #         plt.scatter(bo_data['volumetric'], bo_data['max_cumulative_pressure'],
# #                     color='blue', alpha=0.7, s=80, label=f'BO ({len(bo_data)})', marker='o')
# #
# #     if not nsga3_data.empty:
# #         plt.scatter(nsga3_data['volumetric'], nsga3_data['max_cumulative_pressure'],
# #                     color='orange', alpha=0.7, s=80, label=f'NSGA-III ({len(nsga3_data)})', marker='^')
# #
# #     plt.xlabel('Volumetric Loss [W]', fontsize=16)
# #     plt.ylabel('Max Cumulative Contact Pressure [Pa]', fontsize=16)
# #     plt.title('Contact Pressure vs Volumetric Loss', fontsize=14)
# #     plt.xscale('log')
# #     plt.yscale('log')
# #     plt.grid(True, alpha=0.3)
# #     plt.legend(fontsize=14)
# #     plt.xticks(fontsize=14)
# #     plt.yticks(fontsize=14)
# #     plt.tight_layout()
# #     plt.savefig(output_dir / "contact_pressure_vs_volumetric_loss.png", dpi=300)
# #     plt.close()
# #
# #     # 3. Contact pressure vs mechanical loss
# #     plt.figure(figsize=(8, 6))
# #
# #     if not bo_data.empty:
# #         plt.scatter(bo_data['mechanical'], bo_data['mean_cumulative_pressure'],
# #                     color='blue', alpha=0.7, s=80, label=f'BO ({len(bo_data)})', marker='o')
# #
# #     if not nsga3_data.empty:
# #         plt.scatter(nsga3_data['mechanical'], nsga3_data['mean_cumulative_pressure'],
# #                     color='orange', alpha=0.7, s=80, label=f'NSGA-III ({len(nsga3_data)})', marker='^')
# #
# #     plt.xlabel('Mechanical Loss [W]', fontsize=16)
# #     plt.ylabel('Mean Cumulative Contact Pressure [Pa]', fontsize=16)
# #     plt.title('Contact Pressure vs Mechanical Loss', fontsize=14)
# #     plt.xscale('log')
# #     plt.yscale('log')
# #     plt.grid(True, alpha=0.3)
# #     plt.legend(fontsize=14)
# #     plt.xticks(fontsize=14)
# #     plt.yticks(fontsize=14)
# #     plt.tight_layout()
# #     plt.savefig(output_dir / "contact_pressure_vs_mechanical_loss.png", dpi=300)
# #     plt.close()
# #
# #     # 4. Contact pressure vs total loss
# #     plt.figure(figsize=(8, 6))
# #
# #     if not bo_data.empty:
# #         plt.scatter(bo_data['total'], bo_data['mean_cumulative_pressure'],
# #                     color='blue', alpha=0.7, s=80, label=f'BO ({len(bo_data)})', marker='o')
# #
# #     if not nsga3_data.empty:
# #         plt.scatter(nsga3_data['total'], nsga3_data['mean_cumulative_pressure'],
# #                     color='orange', alpha=0.7, s=80, label=f'NSGA-III ({len(nsga3_data)})', marker='^')
# #
# #     plt.xlabel('Total Loss [W]', fontsize=16)
# #     plt.ylabel('Mean Cumulative Contact Pressure [Pa]', fontsize=16)
# #     plt.title('Contact Pressure vs Total Loss', fontsize=14)
# #     plt.xscale('log')
# #     plt.yscale('log')
# #     plt.grid(True, alpha=0.3)
# #     plt.legend(fontsize=14)
# #     plt.xticks(fontsize=14)
# #     plt.yticks(fontsize=14)
# #     plt.tight_layout()
# #     plt.savefig(output_dir / "contact_pressure_vs_total_loss.png", dpi=300)
# #     plt.close()
# #
# #     # 5. Summary statistics table as image
# #     if len(contact_results) > 0:
# #         fig, ax = plt.subplots(figsize=(12, 6))
# #         ax.axis('tight')
# #         ax.axis('off')
# #
# #         # Create summary data
# #         summary_data = []
# #         for i, result in enumerate(contact_results):
# #             summary_data.append([
# #                 f"{result['optimizer']}-{i + 1}",
# #                 f"{result['max_cumulative_pressure']:.2e}",
# #                 f"{result['mean_cumulative_pressure']:.2e}",
# #                 f"{result['total']:.2e}",
# #                 f"{result['mechanical']:.2e}",
# #                 f"{result['volumetric']:.2e}",
# #                 f"{result['dK']:.2f}",
# #                 f"{result['dZ']:.2f}",
# #                 f"{result['LKG']:.1f}",
# #                 f"{result['lF']:.1f}",
# #                 f"{result['zeta']:.0f}"
# #             ])
# #
# #         columns = ['ID', 'Max CP [Pa]', 'Mean CP [Pa]', 'Total Loss [W]',
# #                    'Mech Loss [W]', 'Vol Loss [W]', 'dK [mm]', 'dZ [mm]',
# #                    'LKG [mm]', 'lF [mm]', 'ζ']
# #
# #         table = ax.table(cellText=summary_data, colLabels=columns,
# #                          cellLoc='center', loc='center')
# #         table.auto_set_font_size(False)
# #         table.set_fontsize(9)
# #         table.scale(1.2, 1.5)
# #
# #         # Color header
# #         for i in range(len(columns)):
# #             table[(0, i)].set_facecolor('#4CAF50')
# #             table[(0, i)].set_text_props(weight='bold', color='white')
# #
# #         plt.title('Contact Pressure Analysis Summary', fontsize=16, pad=20)
# #         plt.savefig(output_dir / "contact_pressure_summary_table.png", dpi=300, bbox_inches='tight')
# #         plt.close()
# #
# #     print(f"✅ Contact pressure plots saved to {output_dir}")
# #     print(f"   - Generated {4 + (1 if len(contact_results) > 0 else 0)} plots successfully")
#
# def create_contact_pressure_plots(contact_results, output_dir):
#     """
#     Create plots showing contact pressure analysis results.
#     """
#     if not contact_results:
#         print("❌ No contact pressure results to plot - all simulations were skipped due to missing files")
#         print("\n💡 To fix this issue:")
#         print("   1. Ensure the simulation data files exist in the expected locations")
#         print("   2. Check that simulations have completed successfully")
#         print("   3. Verify the file paths in the folder structure")
#         return
#
#     print(f"\n📊 Creating contact pressure plots for {len(contact_results)} successful simulations...")
#
#     df_contact = pd.DataFrame(contact_results)
#
#     # 1. Mean cumulative contact pressure comparison
#     plt.figure(figsize=(12, 6))
#
#     # Separate by optimizer
#     bo_data = df_contact[df_contact['optimizer'] == 'BO']
#     nsga3_data = df_contact[df_contact['optimizer'] == 'NSGA-III']
#
#     # Create positions for bars
#     x_positions = []
#     labels = []
#     colors = []
#     pressures = []
#
#     # Add BO data
#     for i, (_, row) in enumerate(bo_data.iterrows()):
#         x_positions.append(i)
#         labels.append(f"BO-{i + 1}")
#         colors.append('blue')
#         pressures.append(row['max_cumulative_pressure'])
#
#     # Add NSGA-III data
#     nsga_start = len(bo_data)
#     for i, (_, row) in enumerate(nsga3_data.iterrows()):
#         x_positions.append(nsga_start + i)
#         labels.append(f"NSGA-{i + 1}")
#         colors.append('orange')
#         pressures.append(row['max_cumulative_pressure'])
#
#     if pressures:  # Only create plot if we have data
#         plt.bar(x_positions, pressures, color=colors, alpha=0.7, width=0.8)
#         plt.xlabel('Simulation', fontsize=16)
#         plt.ylabel('Max Cumulative Contact Pressure [Pa]', fontsize=16)
#         plt.title(
#             f'Max Cumulative Contact Pressure for Best Optimization Results\n({len(bo_data)} BO + {len(nsga3_data)} NSGA-III)',
#             fontsize=14)
#         plt.yscale('log')
#         plt.xticks(x_positions, labels, rotation=45)
#
#         # Add legend
#         bo_patch = plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label=f'BO ({len(bo_data)})')
#         nsga3_patch = plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.7, label=f'NSGA-III ({len(nsga3_data)})')
#         plt.legend(handles=[bo_patch, nsga3_patch], fontsize=14)
#
#         plt.grid(True, alpha=0.3)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=14)
#         plt.tight_layout()
#         plt.savefig(output_dir / "contact_pressure_comparison.png", dpi=300)
#         plt.close()
#
#     # 2. Contact pressure vs volumetric loss (similar to pareto plot style)
#     plt.figure(figsize=(8, 6))
#
#     if not bo_data.empty:
#         plt.scatter(bo_data['volumetric'], bo_data['max_cumulative_pressure'],
#                     color='blue', alpha=0.7, s=80, label=f'BO ({len(bo_data)})', marker='o')
#
#     if not nsga3_data.empty:
#         plt.scatter(nsga3_data['volumetric'], nsga3_data['max_cumulative_pressure'],
#                     color='orange', alpha=0.7, s=80, label=f'NSGA-III ({len(nsga3_data)})', marker='^')
#
#     plt.xlabel('Volumetric Loss [W]', fontsize=16)
#     plt.ylabel('Max Cumulative Contact Pressure [Pa]', fontsize=16)
#     plt.title('Contact Pressure vs Volumetric Loss', fontsize=14)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig(output_dir / "contact_pressure_vs_volumetric_loss.png", dpi=300)
#     plt.close()
#
#     # 3. Contact pressure vs mechanical loss
#     plt.figure(figsize=(8, 6))
#
#     if not bo_data.empty:
#         plt.scatter(bo_data['mechanical'], bo_data['mean_cumulative_pressure'],
#                     color='blue', alpha=0.7, s=80, label=f'BO ({len(bo_data)})', marker='o')
#
#     if not nsga3_data.empty:
#         plt.scatter(nsga3_data['mechanical'], nsga3_data['mean_cumulative_pressure'],
#                     color='orange', alpha=0.7, s=80, label=f'NSGA-III ({len(nsga3_data)})', marker='^')
#
#     plt.xlabel('Mechanical Loss [W]', fontsize=16)
#     plt.ylabel('Mean Cumulative Contact Pressure [Pa]', fontsize=16)
#     plt.title('Contact Pressure vs Mechanical Loss', fontsize=14)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig(output_dir / "contact_pressure_vs_mechanical_loss.png", dpi=300)
#     plt.close()
#
#     # 4. Contact pressure vs total loss
#     plt.figure(figsize=(8, 6))
#
#     if not bo_data.empty:
#         plt.scatter(bo_data['total'], bo_data['mean_cumulative_pressure'],
#                     color='blue', alpha=0.7, s=80, label=f'BO ({len(bo_data)})', marker='o')
#
#     if not nsga3_data.empty:
#         plt.scatter(nsga3_data['total'], nsga3_data['mean_cumulative_pressure'],
#                     color='orange', alpha=0.7, s=80, label=f'NSGA-III ({len(nsga3_data)})', marker='^')
#
#     plt.xlabel('Total Loss [W]', fontsize=16)
#     plt.ylabel('Mean Cumulative Contact Pressure [Pa]', fontsize=16)
#     plt.title('Contact Pressure vs Total Loss', fontsize=14)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig(output_dir / "contact_pressure_vs_total_loss.png", dpi=300)
#     plt.close()
#
#     # 5. Summary statistics table as image
#     if len(contact_results) > 0:
#         fig, ax = plt.subplots(figsize=(12, 6))
#         ax.axis('tight')
#         ax.axis('off')
#
#         # Create summary data
#         summary_data = []
#         for i, result in enumerate(contact_results):
#             summary_data.append([
#                 f"{result['optimizer']}-{i + 1}",
#                 f"{result['max_cumulative_pressure']:.2e}",
#                 f"{result['mean_cumulative_pressure']:.2e}",
#                 f"{result['total']:.2e}",
#                 f"{result['mechanical']:.2e}",
#                 f"{result['volumetric']:.2e}",
#                 f"{result['dK']:.2f}",
#                 f"{result['dZ']:.2f}",
#                 f"{result['LKG']:.1f}",
#                 f"{result['lF']:.1f}",
#                 f"{result['zeta']:.0f}"
#             ])
#
#         columns = ['ID', 'Max CP [Pa]', 'Mean CP [Pa]', 'Total Loss [W]',
#                    'Mech Loss [W]', 'Vol Loss [W]', 'dK [mm]', 'dZ [mm]',
#                    'LKG [mm]', 'lF [mm]', 'ζ']
#
#         table = ax.table(cellText=summary_data, colLabels=columns,
#                          cellLoc='center', loc='center')
#         table.auto_set_font_size(False)
#         table.set_fontsize(9)
#         table.scale(1.2, 1.5)
#
#         # Color header
#         for i in range(len(columns)):
#             table[(0, i)].set_facecolor('#4CAF50')
#             table[(0, i)].set_text_props(weight='bold', color='white')
#
#         plt.title('Contact Pressure Analysis Summary', fontsize=16, pad=20)
#         plt.savefig(output_dir / "contact_pressure_summary_table.png", dpi=300, bbox_inches='tight')
#         plt.close()
#
#     # 6. NEW PLOT: Contact Pressure vs Total Loss for All Best Designs
#     plt.figure(figsize=(8, 6))
#
#     if not bo_data.empty:
#         plt.scatter(bo_data['total'], bo_data['max_cumulative_pressure'],
#                     color='blue', alpha=0.7, s=80, label=f'BO ({len(bo_data)})', marker='o')
#
#     if not nsga3_data.empty:
#         plt.scatter(nsga3_data['total'], nsga3_data['max_cumulative_pressure'],
#                     color='orange', alpha=0.7, s=80, label=f'NSGA-III ({len(nsga3_data)})', marker='^')
#
#     plt.xlabel('Total Loss [W]', fontsize=16)
#     plt.ylabel('Max Cumulative Contact Pressure [Pa]', fontsize=16)
#     plt.title('Contact Pressure vs Total Loss for All Best Designs', fontsize=14)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()
#     plt.savefig(output_dir / "combined_contact_pressure_vs_total_loss_all_best.png", dpi=300)
#     plt.close()
#
#     print(f"✅ Contact pressure plots saved to {output_dir}")
#     print(f"   - Generated {5 + (1 if len(contact_results) > 0 else 0)} plots successfully")
#     print(f"   - NEW: Combined contact pressure vs total loss plot added")
#
# def main():
#     """
#     Main function that runs the complete integrated analysis.
#     """
#
#     # # --- Load Data for Both Optimizers ---
#     # bo_folder = 'Z:/Studenten/Mit/Inline_Thesis-Simulation/HSP/RUN/Run_optimizer_test/Run1_length_diameter_zeta_optimization_BO/bayesian_optimization'
#     # nsga3_folder = 'Z:/Studenten/Mit/Inline_Thesis-Simulation/HSP/RUN/Run_optimizer_test/Run4_length_diameter_zeta_optimization_simple_nsga_more_generation/advanced_nsga3'
#     # bo_folder= r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run7_diameter_psiton_bushing_length_piston_BO\bayesian_optimization_20250726_182402'
#     # nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run6_diameter_psiton_bushing_length_piston_advance_nsga\advanced_nsga3'
#     # Use raw strings or forward slashes for Windows paths
#     bo_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run6_diameter_psiton_bushing_length_piston_advance_nsga\bayesian_optimization'
#     # nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run5_diameter_zeta_simple_nsga-III\simple_nsga3'
#     nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run6_diameter_psiton_bushing_length_piston_advance_nsga\advanced_nsga3'
#     bo_results = load_results(bo_folder, "BO")
#     nsga3_results = load_results(nsga3_folder, "NSGA-III")
#
#     # Convert to DataFrame for convenience
#     df_bo = pd.DataFrame(bo_results)
#     df_nsga3 = pd.DataFrame(nsga3_results)
#
#     # Ensure a common iteration column name for both
#     if 'generation' in df_nsga3.columns:
#         df_nsga3 = df_nsga3.rename(columns={'generation': 'iteration'})
#
#     # Create output directory for plots
#     output_dir = Path("chatgpt_3_para/optimization_plots")
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     # Define parameter names for iteration
#     parameters = ["dK", "dZ", "LKG", "lF", "zeta"]
#
#     # --- ALL ORIGINAL T2 PLOTTING CODE ---
#
#     # 1. Convergence Plot for BO
#     plt.figure(figsize=(8, 5))
#     # Compute best-so-far minimum total loss at each iteration
#     if not df_bo.empty:
#         df_bo_best = df_bo.groupby('iteration')['total'].min().reset_index()
#         plt.plot(df_bo_best['iteration'], df_bo_best['total'], marker='o', label='BO')
#     plt.title("BO Convergence: Best Total Loss vs Iteration")
#     plt.xlabel("Iteration")
#     plt.ylabel("Best Total Loss (min up to iteration)")
#     plt.yscale('log')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(output_dir / "convergence_BO.png")
#     plt.close()
#
#     # 1. Convergence Plot for NSGA-III
#     plt.figure(figsize=(8, 5))
#     if not df_nsga3.empty:
#         df_nsga3_best = df_nsga3.groupby('iteration')['total'].min().reset_index()
#         plt.plot(df_nsga3_best['iteration'], df_nsga3_best['total'], color='orange', marker='o', label='NSGA-III')
#     # plt.title("NSGA-III Convergence: Best Total Loss vs Generation")
#     plt.xlabel("Generation")
#     plt.ylabel("Best Total Loss [W]")
#     plt.yscale('log')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(output_dir / "convergence_NSGA3.png")
#     plt.close()
#
#     # Replace the combined parameter evolution section in your main() function with this code:
#
#     # --- Combined Parameter Evolution Plot with Clearance and Convergence ---
#     df_bo_best_iter_combined = pd.DataFrame()
#     if not df_bo.empty:
#         df_bo_best_iter_combined = df_bo.loc[df_bo.groupby('iteration')['total'].idxmin()].copy()
#         # Calculate clearance for BO
#         df_bo_best_iter_combined['clearance'] = df_bo_best_iter_combined['dZ'] - df_bo_best_iter_combined['dK']
#
#     df_nsga3_best_iter_combined = pd.DataFrame()
#     if not df_nsga3.empty:
#         df_nsga3_best_iter_combined = df_nsga3.loc[df_nsga3.groupby('iteration')['total'].idxmin()].copy()
#         # Calculate clearance for NSGA-III
#         df_nsga3_best_iter_combined['clearance'] = df_nsga3_best_iter_combined['dZ'] - df_nsga3_best_iter_combined['dK']
#
#     # Update parameters list to include clearance instead of dK and dZ, plus convergence
#     parameters_with_clearance = ["clearance", "LKG", "lF", "zeta"]
#
#     # Add convergence plot to the plots (total power loss evolution)
#     plots_to_show = parameters_with_clearance + ["convergence"]
#
#     if not df_bo_best_iter_combined.empty or not df_nsga3_best_iter_combined.empty:
#         fig, axes = plt.subplots(len(plots_to_show), 1, figsize=(10, 18), sharex=True)
#         if len(plots_to_show) == 1:
#             axes = [axes]
#
#         for i, param in enumerate(plots_to_show):
#             if param == "convergence":
#                 # Special handling for convergence plot
#                 if not df_bo.empty:
#                     df_bo_conv = df_bo.groupby('iteration')['total'].min().reset_index()
#                     axes[i].plot(df_bo_conv['iteration'], df_bo_conv['total'],
#                                  marker='o', linestyle='-', color='blue', label='BO', linewidth=2)
#
#                 if not df_nsga3.empty:
#                     df_nsga3_conv = df_nsga3.groupby('iteration')['total'].min().reset_index()
#                     axes[i].plot(df_nsga3_conv['iteration'], df_nsga3_conv['total'],
#                                  marker='x', linestyle='--', color='orange', label='NSGA-III', linewidth=2)
#
#                 axes[i].set_ylabel("Best Total Power Loss [W]", fontsize=16)
#                 axes[i].set_yscale('log')
#
#             else:
#                 # Regular parameter plots
#                 if not df_bo_best_iter_combined.empty:
#                     axes[i].plot(df_bo_best_iter_combined['iteration'], df_bo_best_iter_combined[param],
#                                  marker='o', linestyle='-', color='blue', label='BO')
#                 if not df_nsga3_best_iter_combined.empty:
#                     axes[i].plot(df_nsga3_best_iter_combined['iteration'], df_nsga3_best_iter_combined[param],
#                                  marker='x', linestyle='--', color='orange', label='NSGA-III')
#
#                 # Set proper labels for parameters
#                 if param == "zeta":
#                     y_label = "γ"
#                 elif param == "clearance":
#                     y_label = "clearance [um]"
#                 elif param == "LKG":
#                     y_label = "LKG [mm]"
#                 elif param == "lF":
#                     y_label = "lF [mm]"
#                 elif param == "convergence":
#                     y_label = "Best Total Power Loss [W]"
#
#                 else:
#                     y_label = param
#
#                 axes[i].set_ylabel(y_label, fontsize=16)
#
#                 # Calculate y-limits for parameters
#                 all_param_values = []
#                 if not df_bo_best_iter_combined.empty and param in df_bo_best_iter_combined.columns:
#                     all_param_values.extend(df_bo_best_iter_combined[param].tolist())
#                 if not df_nsga3_best_iter_combined.empty and param in df_nsga3_best_iter_combined.columns:
#                     all_param_values.extend(df_nsga3_best_iter_combined[param].tolist())
#
#                 if all_param_values:
#                     min_val = min(all_param_values)
#                     max_val = max(all_param_values)
#                     if min_val == max_val:
#                         pad = 0.1 * abs(min_val) if min_val != 0 else 0.1
#                         axes[i].set_ylim(min_val - pad, max_val + pad)
#                     else:
#                         range_val = max_val - min_val
#                         axes[i].set_ylim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)
#
#             axes[i].tick_params(axis='both', labelsize=16)
#             axes[i].grid(True)
#             axes[i].legend(fontsize=18)
#
#         axes[-1].set_xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
#         axes[-1].tick_params(axis='both', labelsize=16)
#
#         fig.tight_layout()
#         fig.savefig(output_dir / "combined_param_evolution_with_clearance_and_convergence.png", dpi=300)
#         plt.close(fig)
#
#     # 3. Parameter Distribution Histograms for BO
#     if not df_bo.empty:
#         fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
#         for i, param in enumerate(parameters):
#             sns.histplot(df_bo, x=param, kde=True, ax=axes[i], color='blue', bins=20)
#             axes[i].set_title(f"BO: Distribution of {param}")
#             axes[i].set_xlabel(param)
#             axes[i].set_ylabel("Count")
#         fig.tight_layout()
#         fig.savefig(output_dir / "param_distribution_BO.png")
#         plt.close(fig)
#
#     # 3. Parameter Distribution Histograms for NSGA-III
#     if not df_nsga3.empty:
#         fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
#         for i, param in enumerate(parameters):
#             sns.histplot(df_nsga3, x=param, kde=True, ax=axes[i], color='orange', bins=20)
#             axes[i].set_title(f"NSGA-III: Distribution of {param}")
#             axes[i].set_xlabel(param)
#             axes[i].set_ylabel("Count")
#         fig.tight_layout()
#         fig.savefig(output_dir / "param_distribution_NSGA3.png")
#         plt.close(fig)
#
#     # 4. Parameter vs Total Loss Scatter Plots for BO
#     if not df_bo.empty:
#         fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
#         for i, param in enumerate(parameters):
#             axes[i].scatter(df_bo[param], df_bo["total"], color='blue', alpha=0.6)
#             axes[i].set_title(f"BO: {param} vs Total Loss")
#             axes[i].set_xlabel(param)
#             axes[i].set_ylabel("Total Loss")
#             axes[i].set_yscale('log')
#             axes[i].grid(True)
#         fig.tight_layout()
#         fig.savefig(output_dir / "param_vs_loss_BO.png")
#         plt.close(fig)
#
#     # 4. Parameter vs Total Loss Scatter Plots for NSGA-III
#     if not df_nsga3.empty:
#         fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
#         for i, param in enumerate(parameters):
#             axes[i].scatter(df_nsga3[param], df_nsga3["total"], color='orange', alpha=0.6)
#             axes[i].set_title(f"NSGA-III: {param} vs Total Loss")
#             axes[i].set_xlabel(param)
#             axes[i].set_ylabel("Total Loss")
#             axes[i].set_yscale('log')
#             axes[i].grid(True)
#         fig.tight_layout()
#         fig.savefig(output_dir / "param_vs_loss_NSGA3.png")
#         plt.close(fig)
#
#     # 5. Pair Plot for BO (parameters vs parameters and vs total)
#     if not df_bo.empty:
#         cols_for_pairplot = parameters + ["total"]
#         # Use a sample if there are too many points for clarity
#         df_plot = df_bo.copy()
#         if len(df_plot) > 5000:
#             df_plot = df_plot.sample(n=5000, random_state=42)
#         sns.pairplot(df_plot[cols_for_pairplot], diag_kind="kde")
#         plt.suptitle("BO Pairwise Parameter Relationships", y=1.02)
#         plt.savefig(output_dir / "pairplot_BO.png")
#         plt.close()  # close the whole pairplot figure
#
#     # 5. Pair Plot for NSGA-III
#     if not df_nsga3.empty:
#         cols_for_pairplot = parameters + ["total"]
#         df_plot = df_nsga3.copy()
#         if len(df_plot) > 5000:
#             df_plot = df_plot.sample(n=5000, random_state=42)
#         sns.pairplot(df_plot[cols_for_pairplot], diag_kind="kde")
#         plt.suptitle("NSGA-III Pairwise Parameter Relationships", y=1.02)
#         plt.savefig(output_dir / "pairplot_NSGA3.png")
#         plt.close()
#
#     # 6. Pareto Front Scatter (Mechanical vs Volumetric Loss) for BO
#     if not df_bo.empty and "mechanical" in df_bo.columns and "volumetric" in df_bo.columns:
#         plt.figure(figsize=(6, 5))
#         plt.scatter(df_bo["mechanical"], df_bo["volumetric"], color='blue', alpha=0.7)
#         plt.title("BO: Mechanical vs Volumetric Loss")
#         plt.xlabel("Mechanical Loss")
#         plt.ylabel("Volumetric Loss")
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.grid(True)
#         plt.savefig(output_dir / "pareto_BO.png")
#         plt.close()
#
#     # 6. Pareto Front Scatter for NSGA-III
#     if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
#         plt.figure(figsize=(6, 5))
#         plt.scatter(df_nsga3["mechanical"], df_nsga3["volumetric"], color='orange', alpha=0.7)
#         plt.title("NSGA-III: Mechanical vs Volumetric Loss")
#         plt.xlabel("Mechanical Loss")
#         plt.ylabel("Volumetric Loss")
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.grid(True)
#         plt.savefig(output_dir / "pareto_NSGA3.png")
#         plt.close()
#
#     print(f"Original T2 plots saved to: {output_dir}")
#
#     # 7. Save Top 5 Optimal Results (by Total Loss) for Each Optimizer
#     top_n_save = 5
#     columns_to_save = ["optimizer", "iteration", "total", "mechanical", "volumetric", "dK", "dZ", "LKG", "lF", "zeta"]
#
#     # Get top 5 BO results
#     top_bo_save = pd.DataFrame()
#     if not df_bo.empty:
#         top_bo_save = df_bo.nsmallest(top_n_save, 'total').copy()
#         top_bo_save["optimizer"] = "BO"
#         top_bo_save = top_bo_save[columns_to_save]
#
#     # Get top 5 NSGA-III results
#     top_nsga3_save = pd.DataFrame()
#     if not df_nsga3.empty:
#         top_nsga3_save = df_nsga3.nsmallest(top_n_save, 'total').copy()
#         top_nsga3_save["optimizer"] = "NSGA-III"
#         top_nsga3_save = top_nsga3_save[columns_to_save]
#
#     # Combine and save
#     df_top_combined = pd.concat([top_bo_save, top_nsga3_save], ignore_index=True)
#     df_top_combined.to_csv(output_dir / "top5_optimal_results.csv", index=False)
#
#     print(f"Saved top 5 parameter sets to: {output_dir / 'top5_optimal_results.csv'}")
#
#     # --- Combined Parameter Evolution Plot ---
#     df_bo_best_iter_combined = pd.DataFrame()
#     if not df_bo.empty:
#         df_bo_best_iter_combined = df_bo.loc[df_bo.groupby('iteration')['total'].idxmin()]
#
#     df_nsga3_best_iter_combined = pd.DataFrame()
#     if not df_nsga3.empty:
#         df_nsga3_best_iter_combined = df_nsga3.loc[df_nsga3.groupby('iteration')['total'].idxmin()]
#
#     if not df_bo_best_iter_combined.empty or not df_nsga3_best_iter_combined.empty:
#         fig, axes = plt.subplots(len(parameters), 1, figsize=(10, 15), sharex=True)
#         if len(parameters) == 1:
#             axes = [axes]
#
#         for i, param in enumerate(parameters):
#             if not df_bo_best_iter_combined.empty:
#                 axes[i].plot(df_bo_best_iter_combined['iteration'], df_bo_best_iter_combined[param],
#                              marker='o', linestyle='-', color='blue', label='BO')
#             if not df_nsga3_best_iter_combined.empty:
#                 axes[i].plot(df_nsga3_best_iter_combined['iteration'], df_nsga3_best_iter_combined[param],
#                              marker='x', linestyle='--', color='orange', label='NSGA-III')
#
#             y_label = "γ" if param == "zeta" else param
#             axes[i].set_ylabel(y_label, fontsize=16)
#
#             all_param_values = []
#             if not df_bo_best_iter_combined.empty:
#                 all_param_values.extend(df_bo_best_iter_combined[param].tolist())
#             if not df_nsga3_best_iter_combined.empty:
#                 all_param_values.extend(df_nsga3_best_iter_combined[param].tolist())
#             if all_param_values:
#                 min_val = min(all_param_values)
#                 max_val = max(all_param_values)
#                 if min_val == max_val:
#                     pad = 0.1 * abs(min_val) if min_val != 0 else 0.1
#                     axes[i].set_ylim(min_val - pad, max_val + pad)
#                 else:
#                     range_val = max_val - min_val
#                     axes[i].set_ylim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)
#
#             axes[i].tick_params(axis='both', labelsize=16)
#             axes[i].grid(True)
#             axes[i].legend(fontsize=18)
#
#         axes[-1].set_xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
#         axes[-1].tick_params(axis='both', labelsize=16)
#
#         fig.tight_layout()
#         fig.savefig(output_dir / "combined_param_evolution.png")
#         plt.close(fig)
#
#     # --- Combined Convergence Plot ---
#     plt.figure(figsize=(9, 6))
#     plot_data_exists = False
#
#     if not df_bo.empty:
#         df_bo_best_conv = df_bo.groupby('iteration')['total'].min().reset_index()
#         plt.plot(df_bo_best_conv['iteration'], df_bo_best_conv['total'], marker='o', linestyle='-', color='blue',
#                  label='BO')
#         plot_data_exists = True
#
#     if not df_nsga3.empty:
#         df_nsga3_best_conv = df_nsga3.groupby('iteration')['total'].min().reset_index()
#         plt.plot(df_nsga3_best_conv['iteration'], df_nsga3_best_conv['total'], marker='x', linestyle='--',
#                  color='orange', label='NSGA-III')
#         plot_data_exists = True
#
#     if plot_data_exists:
#         plt.xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
#         plt.ylabel("Best Total Loss [W]", fontsize=16)
#         plt.yscale('log')
#         plt.grid(True)
#         plt.legend(fontsize=18)
#         plt.xticks(fontsize=16)
#         plt.yticks(fontsize=16)
#         plt.tight_layout()
#         plt.savefig(output_dir / "combined_convergence.png")
#         plt.close()
#     else:
#         print("No data available to generate combined convergence plot.")
#
#     # --- Combined Pareto Front Plot ---
#     if not df_bo.empty or not df_nsga3.empty:
#         plt.figure(figsize=(7, 6))
#
#         if not df_bo.empty and "mechanical" in df_bo.columns and "volumetric" in df_bo.columns:
#             plt.scatter(df_bo["mechanical"], df_bo["volumetric"], color='blue', alpha=0.6, label='BO')
#
#         if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
#             plt.scatter(df_nsga3["mechanical"], df_nsga3["volumetric"], color='orange', alpha=0.6, label='NSGA-III')
#
#         plt.xlabel("Mechanical Loss[W]", fontsize=16)
#         plt.ylabel("Volumetric Loss[W]", fontsize=16)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xticks(fontsize=20)
#         plt.yticks(fontsize=20)
#         plt.grid(True)
#         plt.legend(fontsize=20)
#         plt.tight_layout()
#         plt.savefig(output_dir / "combined_pareto.png")
#         plt.close()
#
#     # --- Sensitivity Bar Chart ---
#     # Compute clearance
#     df_bo["clearance"] = df_bo["dZ"] - df_bo["dK"]
#     df_nsga3["clearance"] = df_nsga3["dZ"] - df_nsga3["dK"]
#
#     # Parameters to analyze
#     selected_params = ["clearance", "zeta", "lF", "LKG"]
#     # selected_params = ["clearance", "zeta"]
#     # Human-readable labels
#     param_labels = {
#         "clearance": "clearance",
#         "zeta": "γ",
#         "lF": "lF",
#         "LKG": "LKG"
#     }
#
#     sensitivity_data = []
#
#     for param in selected_params:
#         bo_corr = np.nan
#         nsga_corr = np.nan
#
#         if not df_bo.empty and param in df_bo.columns:
#             bo_corr, _ = pearsonr(df_bo[param], df_bo["total"])
#             bo_corr = abs(bo_corr)
#
#         if not df_nsga3.empty and param in df_nsga3.columns:
#             nsga_corr, _ = pearsonr(df_nsga3[param], df_nsga3["total"])
#             nsga_corr = abs(nsga_corr)
#
#         sensitivity_data.append({
#             "parameter": param_labels[param],
#             "BO": bo_corr,
#             "NSGA-III": nsga_corr
#         })
#
#     # Convert to DataFrame
#     df_sensitivity = pd.DataFrame(sensitivity_data)
#
#     # Plotting
#     x = np.arange(len(df_sensitivity["parameter"]))
#     width = 0.35
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#     bars1 = ax.bar(x - width / 2, df_sensitivity["BO"], width, label='BO', color='blue')
#     bars2 = ax.bar(x + width / 2, df_sensitivity["NSGA-III"], width, label='NSGA-III', color='orange')
#
#     ax.set_xlabel("Parameter", fontsize=16)
#     ax.set_ylabel("Sensitivity (|Pearson Correlation|)", fontsize=16)
#     ax.set_xticks(x)
#     ax.set_xticklabels(df_sensitivity["parameter"], fontsize=16)
#     ax.tick_params(axis='y', labelsize=16)
#     ax.legend(fontsize=18)
#     ax.grid(True, axis='y')
#     fig.tight_layout()
#     fig.savefig(output_dir / "sensitivity_bar_chart_all_params.png")
#     plt.close(fig)
#
#     print("\n--- Top 10 Optimal Results from Pareto Front (NSGA-III) and Lowest Total Loss (BO) ---")
#
#     top_n_display = 10
#     columns_to_show = ["optimizer", "total", "mechanical", "volumetric", "dK", "dZ", "LKG", "lF", "zeta", "folder_name"]
#
#     # --- For BO: Get top N by total loss ---
#     top_bo_results = pd.DataFrame()
#     if not df_bo.empty:
#         top_bo_results = df_bo.nsmallest(top_n_display, 'total').copy()
#         top_bo_results["Optimizer Selection"] = "Lowest Total Loss"
#         top_bo_results = top_bo_results[columns_to_show]
#         print("\nBO Top {} (by lowest total loss):".format(len(top_bo_results)))
#         if not top_bo_results.empty:
#             for i, row in top_bo_results.iterrows():
#                 print(
#                     f"{i + 1}. Total Loss: {row['total']:.4e}, Mech Loss: {row['mechanical']:.4e}, Vol Loss: {row['volumetric']:.4e}")
#                 print(
#                     f"    Parameters: dK={row['dK']:.4f}, dZ={row['dZ']:.4f}, LKG={row['LKG']:.4f}, lF={row['lF']:.4f}, zeta={row['zeta']:.0f}")
#                 print(f"    Folder: {row['folder_name']}")
#         else:
#             print("  No valid BO results to display.")
#
#     # --- For NSGA-III: Find Pareto front and then select top N by total loss from Pareto front ---
#     top_nsga3_results = pd.DataFrame()
#     if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
#         pareto_nsga3 = find_pareto_front(df_nsga3, ["mechanical", "volumetric"])
#
#         if not pareto_nsga3.empty:
#             top_nsga3_results = pareto_nsga3.sort_values(by='total').head(top_n_display).copy()
#             top_nsga3_results["Optimizer Selection"] = "Pareto Front (Sorted by Total Loss)"
#             top_nsga3_results = top_nsga3_results[columns_to_show]
#             print(f"\nNSGA-III Top {top_nsga3_results.shape[0]} (from Pareto Front, sorted by total loss):")
#             if not top_nsga3_results.empty:
#                 for i, row in top_nsga3_results.iterrows():
#                     print(
#                         f"{i + 1}. Total Loss: {row['total']:.4e}, Mech Loss: {row['mechanical']:.4e}, Vol Loss: {row['volumetric']:.4e}")
#                     print(
#                         f"    Parameters: dK={row['dK']:.4f}, dZ={row['dZ']:.4f}, LKG={row['LKG']:.4f}, lF={row['lF']:.4f}, zeta={row['zeta']:.0f}")
#                     print(f"    Folder: {row['folder_name']}")
#         else:
#             print("  No Pareto optimal solutions found for NSGA-III.")
#     else:
#         print("  NSGA-III data is empty or missing 'mechanical'/'volumetric' columns.")
#
#     # --- GET TOP 10 RESULTS FROM EACH OPTIMIZER FOR CONTACT PRESSURE ANALYSIS ---
#     print("\n" + "=" * 60)
#     print("SELECTING TOP 10 RESULTS FROM EACH OPTIMIZER FOR CONTACT PRESSURE ANALYSIS")
#     print("=" * 60)
#
#     top_n = 10
#     best_folders = []
#
#     # Get top 10 BO results by total loss
#     if not df_bo.empty:
#         top_bo = df_bo.nsmallest(top_n, 'total').copy()
#         for _, row in top_bo.iterrows():
#             best_folders.append(row.to_dict())
#         print(f"✅ Selected {len(top_bo)} best BO results")
#
#     # Get top 10 NSGA-III results from Pareto front
#     if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
#         pareto_nsga3 = find_pareto_front(df_nsga3, ["mechanical", "volumetric"])
#         if not pareto_nsga3.empty:
#             # Sort Pareto front by total loss and take top 10
#             top_nsga3 = pareto_nsga3.sort_values(by='total').head(top_n).copy()
#             for _, row in top_nsga3.iterrows():
#                 best_folders.append(row.to_dict())
#             print(f"✅ Selected {len(top_nsga3)} best NSGA-III results from Pareto front")
#
#     print(f"Total simulations selected for contact pressure analysis: {len(best_folders)}")
#
#     # --- RUN CONTACT PRESSURE ANALYSIS ---
#     contact_results = run_contact_pressure_for_best_results(best_folders, output_dir)
#
#     # --- CREATE CONTACT PRESSURE PLOTS ---
#     create_contact_pressure_plots(contact_results, output_dir)
#
#     # --- SAVE RESULTS TO CSV ---
#     if contact_results:
#         df_contact_results = pd.DataFrame(contact_results)
#         # Select relevant columns for CSV output
#         csv_columns = ['optimizer', 'analysis_type', 'max_cumulative_pressure', 'mean_cumulative_pressure',
#                        'total', 'mechanical', 'volumetric', 'dK', 'dZ', 'LKG', 'lF', 'zeta', 'folder_name']
#         df_contact_results[csv_columns].to_csv(
#             output_dir / "contact_pressure_results.csv", index=False
#         )
#         print(f"✅ Contact pressure results saved to {output_dir / 'contact_pressure_results.csv'}")
#
#         # Also create a summary of analysis types used
#         analysis_summary = df_contact_results['analysis_type'].value_counts()
#         print(f"📊 Analysis methods used:")
#         for method, count in analysis_summary.items():
#             print(f"   - {method}: {count} simulations")
#
#     else:
#         print("❌ No contact pressure results to save - creating empty results file with headers")
#         # Create empty CSV with headers for reference
#         empty_df = pd.DataFrame(
#             columns=['optimizer', 'analysis_type', 'max_cumulative_pressure', 'mean_cumulative_pressure',
#                      'total', 'mechanical', 'volumetric', 'dK', 'dZ', 'LKG', 'lF', 'zeta', 'folder_name'])
#         empty_df.to_csv(output_dir / "contact_pressure_results_empty.csv", index=False)
#
#         # Save information about missing files for debugging
#         debug_info = []
#         for i, folder_info in enumerate(best_folders):
#             folder_path = folder_info['folder_name']
#             missing_files, available_alternatives = check_required_files(folder_path)
#             debug_info.append({
#                 'folder_index': i + 1,
#                 'optimizer': folder_info['optimizer'],
#                 'folder_name': folder_path,
#                 'missing_file_count': len(missing_files),
#                 'available_alternatives_count': len(available_alternatives),
#                 'first_missing_file': missing_files[0] if missing_files else 'None',
#                 'first_available_file': available_alternatives[0] if available_alternatives else 'None'
#             })
#
#         debug_df = pd.DataFrame(debug_info)
#         debug_df.to_csv(output_dir / "missing_files_debug.csv", index=False)
#         print(f"📋 Debug information saved to {output_dir / 'missing_files_debug.csv'}")
#
#     print("\n--- Process complete. ---")
#
#     print("\n" + "=" * 60)
#     print("INTEGRATED ANALYSIS COMPLETED")
#     print("=" * 60)
#     if contact_results:
#         full_analysis_count = sum(1 for r in contact_results if 'Full T1' in r.get('analysis_type', ''))
#         fallback_count = len(contact_results) - full_analysis_count
#         print(f"✅ SUCCESS: Analyzed {len(contact_results)} simulations with contact pressure data")
#         if full_analysis_count > 0:
#             print(f"   - Full T1 analysis: {full_analysis_count} simulations")
#         if fallback_count > 0:
#             print(f"   - Fallback analysis: {fallback_count} simulations")
#     else:
#         print("⚠️  WARNING: No simulations had complete data for contact pressure analysis")
#         print("   This is likely because:")
#         print("   1. Simulations haven't completed yet")
#         print("   2. Required output files are missing")
#         print("   3. File paths have changed")
#         print("   Check the debug CSV file for more details.")
#     print(f"📁 All results saved to: {output_dir}")
#     print("=" * 60)
#
#
# if __name__ == "__main__":
#     main()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, Optional
import numpy as np
from scipy.stats import pearsonr
import os

# Import all functions from T1
import sys

sys.path.append('..')  # Add current directory to path
from T1 import (
    load_matlab_txt, round2, parse_geometry, parse_operating_conditions,
    create_cumulative_pressure_map, collect_simulation_data,
    create_comparison_plots, create_overall_summary, run_contact_pressure_analysis
)

# --- Updated Data Parsing Functions ---

# Updated regex pattern for new folder naming convention with clearance (CL)
PARAM_PATTERN = re.compile(
    r"CL(?P<CL>[-\d\.]+)_dZ(?P<dZ>[-\d\.]+)_LKG(?P<LKG>[-\d\.]+)_lF(?P<lF>[-\d\.]+)_zeta(?P<zeta>[-\d\.]+)"
)
PARAM_PATTERN = re.compile(
    r"dK(?P<dK>[-\d\.]+)_dZ(?P<dZ>[-\d\.]+)_LKG(?P<LKG>[-\d\.]+)_lF(?P<lF>[-\d\.]+)_zeta(?P<zeta>[-\d\.]+)"
)

def parse_geometry_file(geometry_file_path: str) -> Dict[str, Optional[float]]:
    """
    Extract dK and lF values from geometry.txt file.

    Args:
        geometry_file_path: Path to the geometry.txt file

    Returns:
        Dictionary with dK and lF values, or None values if not found
    """
    result = {"dK": None, "lF": None}

    try:
        if not os.path.exists(geometry_file_path):
            return result

        with open(geometry_file_path, 'r') as file:
            content = file.read()

        # Look for dK parameter in the geometry file
        dk_pattern = re.compile(r'dK\s+([+-]?(?:\d+\.?\d*|\.\d+))')
        dk_match = dk_pattern.search(content)

        # Look for lF parameter in the geometry file
        lf_pattern = re.compile(r'lF\s+([+-]?(?:\d+\.?\d*|\.\d+))')
        lf_match = lf_pattern.search(content)

        if dk_match:
            result["dK"] = float(dk_match.group(1))

        if lf_match:
            result["lF"] = float(lf_match.group(1))

        return result

    except Exception as e:
        print(f"Error reading geometry file {geometry_file_path}: {e}")
        return result


def parse_parameters(folder_name: str, folder_path: str) -> Dict[str, float]:
    """
    Parse parameters from new folder naming convention and extract dK/lF from geometry file.

    Args:
        folder_name: Name of the folder (e.g., "ind2_CL0.028_dZ19.888_LKG56.4_lF36.9_zeta5")
        folder_path: Full path to the folder

    Returns:
        Dictionary containing all parameters including dK extracted from geometry file
    """
    match = PARAM_PATTERN.search(folder_name)
    if not match:
        return {}

    try:
        params = {k: float(v) for k, v in match.groupdict().items()}
        params["zeta"] = float(int(params["zeta"]))  # Keep zeta as integer

        # Extract dK and lF from geometry file
        geometry_file_path = os.path.join(folder_path, 'input', 'geometry.txt')
        geometry_params = parse_geometry_file(geometry_file_path)

        if geometry_params["dK"] is not None:
            params["dK"] = geometry_params["dK"]

            # # Calculate clearance from extracted dK and parsed dZ to verify
            # calculated_clearance = params["dZ"] - params["dK"]
            # parsed_clearance = params["CL"]
            #
            # # Check if calculated clearance matches parsed clearance (with some tolerance)
            # if abs(calculated_clearance - parsed_clearance) > 0.001:  # 1 micron tolerance
            #     print(f"Warning: Clearance mismatch in {folder_name}")
            #     print(f"  Parsed CL: {parsed_clearance:.6f} mm")
            #     print(f"  Calculated (dZ-dK): {calculated_clearance:.6f} mm")
            #
            # # Use the calculated clearance based on extracted dK
            # params["clearance"] = calculated_clearance
        else:
            print(f"Warning: Could not extract dK for {folder_name}, using parsed clearance")
            params["dK"] = params["dZ"] - params["CL"]  # Fallback calculation
            params["clearance"] = params["CL"]

        # Handle lF parameter - prefer geometry file over folder name
        if geometry_params["lF"] is not None:
            geometry_lF = geometry_params["lF"]
            folder_lF = params.get("lF", None)

            # if folder_lF is not None and abs(geometry_lF - folder_lF) > 0.001:
            #     print(f"Info: Using lF from geometry file for {folder_name}: {geometry_lF:.3f} mm")

            params["lF"] = geometry_lF

        return params

    except ValueError as e:
        print(f"Error parsing parameters from {folder_name}: {e}")
        return {}


def parse_loss_file(sim_path: str) -> Dict[str, float]:
    """Parse loss data from piston.txt file"""
    piston_path = Path(sim_path) / "output" / "piston" / "piston.txt"
    if not piston_path.exists():
        return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
    try:
        df_loss = pd.read_csv(piston_path, delimiter="\t")
        df_loss = df_loss[df_loss["revolution"] <= 6.0]
        if df_loss.empty:
            return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
        mech = abs(df_loss["Total_Mechanical_Power_Loss"].max())
        vol = abs(df_loss["Total_Volumetric_Power_Loss"].max())
        if pd.isna(mech) or pd.isna(vol):
            return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}
        return {"mechanical": mech, "volumetric": vol, "total": mech + vol, "valid": True}
    except Exception:
        return {"mechanical": 1e6, "volumetric": 1e6, "total": 2e6, "valid": False}


def load_results(folder_path: str, opt_type: str) -> list:
    """Updated load_results function to use new parameter parsing"""
    base_folder = Path(folder_path)
    results = []
    if not base_folder.exists():
        print(f"Base folder not found: {base_folder}")
        return []

    subfolders = [f for f in base_folder.iterdir() if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders in {base_folder}")

    for folder in subfolders:
        folder_name = folder.name
        if opt_type == "NSGA-III":
            if folder_name.startswith("Generation_G"):
                try:
                    iter_num = int(folder_name.replace("Generation_G", ""))
                    iter_type = "generation"
                except ValueError:
                    continue
            elif folder_name == "Initial_Sampling":
                iter_num = 0
                iter_type = "generation"
            else:
                continue
        elif opt_type == "BO":
            if folder_name.startswith("Iteration_I"):
                try:
                    iter_num = int(folder_name.replace("Iteration_I", ""))
                    iter_type = "iteration"
                except ValueError:
                    continue
            elif folder_name == "Initial_Sampling":
                iter_num = 0
                iter_type = "iteration"
            else:
                continue
        else:
            print(f"Unknown optimization type: {opt_type}")
            continue

        sim_folders = [f for f in folder.iterdir() if f.is_dir()]
        folder_results = 0

        for sim in sim_folders:
            # Updated to use new pattern matching and parameter parsing
            if not PARAM_PATTERN.search(sim.name):
                continue
            params = parse_parameters(sim.name, str(sim))
            if not params:
                continue
            losses = parse_loss_file(str(sim))
            record = {**params, **losses}
            record[iter_type] = iter_num
            record["folder_name"] = str(sim)
            record["optimizer"] = opt_type
            if record.get("valid") and record["total"] < 1e6:
                results.append(record)
                folder_results += 1

        if folder_results > 0:
            print(f"  {folder_name}: {folder_results} valid simulations")

    return results


def find_pareto_front(df: pd.DataFrame, objective_cols: list, minimize: bool = True) -> pd.DataFrame:
    """
    Identifies the Pareto front from a DataFrame of solutions.
    """
    if df.empty or not all(col in df.columns for col in objective_cols):
        return pd.DataFrame()

    df_sorted = df.sort_values(by=objective_cols[0], ascending=minimize).reset_index(drop=True)

    pareto_indices = []
    for i in range(len(df_sorted)):
        is_pareto = True
        for j in range(len(df_sorted)):
            if i == j:
                continue

            dominates_all_objectives = True
            strictly_better_in_one = False

            for obj in objective_cols:
                if minimize:
                    if df_sorted.loc[j, obj] > df_sorted.loc[i, obj]:
                        dominates_all_objectives = False
                        break
                    if df_sorted.loc[j, obj] < df_sorted.loc[i, obj]:
                        strictly_better_in_one = True
                else:
                    if df_sorted.loc[j, obj] < df_sorted.loc[i, obj]:
                        dominates_all_objectives = False
                        break
                    if df_sorted.loc[j, obj] > df_sorted.loc[i, obj]:
                        strictly_better_in_one = True

            if dominates_all_objectives and strictly_better_in_one:
                is_pareto = False
                break

        if is_pareto:
            pareto_indices.append(i)

    return df_sorted.loc[pareto_indices].drop_duplicates().reset_index(drop=True)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def create_enhanced_combined_plots(df_bo, df_nsga3, output_dir):
    """
    Create enhanced combined plots that highlight the best solutions with special colors and annotations.
    """
    # Find the best solutions (lowest total loss) for each optimizer
    best_bo_idx = None
    best_nsga3_idx = None

    if not df_bo.empty:
        best_bo_idx = df_bo['total'].idxmin()
        best_bo_solution = df_bo.loc[best_bo_idx]
        print(
            f"Best BO solution: Total Loss = {best_bo_solution['total']:.2e} W at iteration {best_bo_solution.get('iteration', 'N/A')}")

    if not df_nsga3.empty:
        best_nsga3_idx = df_nsga3['total'].idxmin()
        best_nsga3_solution = df_nsga3.loc[best_nsga3_idx]
        print(
            f"Best NSGA-III solution: Total Loss = {best_nsga3_solution['total']:.2e} W at iteration {best_nsga3_solution.get('iteration', 'N/A')}")

    # 1. Enhanced Combined Convergence Plot
    plt.figure(figsize=(10, 7))
    plot_data_exists = False

    if not df_bo.empty:
        df_bo_best_conv = df_bo.groupby('iteration')['total'].min().reset_index()

        # Plot regular convergence line
        plt.plot(df_bo_best_conv['iteration'], df_bo_best_conv['total'],
                 marker='o', linestyle='-', color='blue', label='BO', linewidth=2, markersize=6)

        # Highlight the best solution point
        if best_bo_idx is not None:
            best_iter = best_bo_solution['iteration']
            best_total = best_bo_solution['total']
            plt.scatter([best_iter], [best_total], color='red', s=150,
                        marker='*', edgecolors='black', linewidth=2,
                        label=f'Best BO (Iter {best_iter})', zorder=5)
            # Add annotation
            plt.annotate(f'Best BO\nIter {best_iter}\n{best_total:.2e} W',
                         xy=(best_iter, best_total), xytext=(10, 20),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plot_data_exists = True

    if not df_nsga3.empty:
        df_nsga3_best_conv = df_nsga3.groupby('iteration')['total'].min().reset_index()

        # Plot regular convergence line
        plt.plot(df_nsga3_best_conv['iteration'], df_nsga3_best_conv['total'],
                 marker='x', linestyle='--', color='orange', label='NSGA-III', linewidth=2, markersize=8)

        # Highlight the best solution point
        if best_nsga3_idx is not None:
            best_iter = best_nsga3_solution['iteration']
            best_total = best_nsga3_solution['total']
            plt.scatter([best_iter], [best_total], color='red', s=150,
                        marker='*', edgecolors='black', linewidth=2,
                        label=f'Best NSGA-III (Gen {best_iter})', zorder=5)
            # Add annotation
            plt.annotate(f'Best NSGA-III\nGen {best_iter}\n{best_total:.2e} W',
                         xy=(best_iter, best_total), xytext=(-10, -30),
                         textcoords='offset points', fontsize=10,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plot_data_exists = True

    if plot_data_exists:
        plt.xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
        plt.ylabel("Best Total Loss [W]", fontsize=16)
        plt.title("Optimization Convergence with Best Solutions Highlighted", fontsize=16)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "enhanced_combined_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Enhanced Combined Pareto Front Plot
    if (not df_bo.empty and "mechanical" in df_bo.columns and "volumetric" in df_bo.columns) or \
            (not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns):

        plt.figure(figsize=(10, 8))

        # Plot regular data points
        if not df_bo.empty and "mechanical" in df_bo.columns and "volumetric" in df_bo.columns:
            plt.scatter(df_bo["mechanical"], df_bo["volumetric"],
                        color='blue', alpha=0.6, s=50, label='BO', marker='o')

        if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
            plt.scatter(df_nsga3["mechanical"], df_nsga3["volumetric"],
                        color='orange', alpha=0.6, s=50, label='NSGA-III', marker='^')

        # Highlight best solutions
        if best_bo_idx is not None and "mechanical" in df_bo.columns:
            best_mech = best_bo_solution["mechanical"]
            best_vol = best_bo_solution["volumetric"]
            best_iter = best_bo_solution["iteration"]
            plt.scatter([best_mech], [best_vol], color='red', s=200,
                        marker='*', edgecolors='black', linewidth=2,
                        label=f'Best BO (Iter {best_iter})', zorder=5)
            plt.annotate(f'Best BO\nIter {best_iter}',
                         xy=(best_mech, best_vol), xytext=(15, 15),
                         textcoords='offset points', fontsize=11,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

        if best_nsga3_idx is not None and "mechanical" in df_nsga3.columns:
            best_mech = best_nsga3_solution["mechanical"]
            best_vol = best_nsga3_solution["volumetric"]
            best_iter = best_nsga3_solution["iteration"]
            plt.scatter([best_mech], [best_vol], color='darkred', s=200,
                        marker='*', edgecolors='black', linewidth=2,
                        label=f'Best NSGA-III (Gen {best_iter})', zorder=5)
            plt.annotate(f'Best NSGA-III\nGen {best_iter}',
                         xy=(best_mech, best_vol), xytext=(-15, -25),
                         textcoords='offset points', fontsize=11,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'))

        plt.xlabel("Mechanical Loss [W]", fontsize=16)
        plt.ylabel("Volumetric Loss [W]", fontsize=16)
        plt.title("Pareto Front with Best Solutions Highlighted", fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "enhanced_combined_pareto.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Enhanced Parameter Evolution Plot with Best Solutions
    parameters_with_clearance = ["clearance", "LKG", "lF", "zeta"]

    # Get best iteration data for both optimizers
    df_bo_best_iter_combined = pd.DataFrame()
    if not df_bo.empty:
        df_bo_best_iter_combined = df_bo.loc[df_bo.groupby('iteration')['total'].idxmin()].copy()
        df_bo_best_iter_combined['clearance'] = df_bo_best_iter_combined['dZ'] - df_bo_best_iter_combined['dK']

    df_nsga3_best_iter_combined = pd.DataFrame()
    if not df_nsga3.empty:
        df_nsga3_best_iter_combined = df_nsga3.loc[df_nsga3.groupby('iteration')['total'].idxmin()].copy()
        df_nsga3_best_iter_combined['clearance'] = df_nsga3_best_iter_combined['dZ'] - df_nsga3_best_iter_combined['dK']

    plots_to_show = parameters_with_clearance + ["convergence"]

    if not df_bo_best_iter_combined.empty or not df_nsga3_best_iter_combined.empty:
        fig, axes = plt.subplots(len(plots_to_show), 1, figsize=(12, 20), sharex=True)
        if len(plots_to_show) == 1:
            axes = [axes]

        for i, param in enumerate(plots_to_show):
            if param == "convergence":
                # Convergence plot with highlighted best solutions
                if not df_bo.empty:
                    df_bo_conv = df_bo.groupby('iteration')['total'].min().reset_index()
                    axes[i].plot(df_bo_conv['iteration'], df_bo_conv['total'],
                                 marker='o', linestyle='-', color='blue', label='BO', linewidth=2)

                    # Highlight best BO solution
                    if best_bo_idx is not None:
                        best_iter = best_bo_solution['iteration']
                        best_total = best_bo_solution['total']
                        axes[i].scatter([best_iter], [best_total], color='red', s=120,
                                        marker='*', edgecolors='black', linewidth=2, zorder=5)

                if not df_nsga3.empty:
                    df_nsga3_conv = df_nsga3.groupby('iteration')['total'].min().reset_index()
                    axes[i].plot(df_nsga3_conv['iteration'], df_nsga3_conv['total'],
                                 marker='x', linestyle='--', color='orange', label='NSGA-III', linewidth=2)

                    # Highlight best NSGA-III solution
                    if best_nsga3_idx is not None:
                        best_iter = best_nsga3_solution['iteration']
                        best_total = best_nsga3_solution['total']
                        axes[i].scatter([best_iter], [best_total], color='darkred', s=120,
                                        marker='*', edgecolors='black', linewidth=2, zorder=5)

                axes[i].set_ylabel("Best Total Power Loss [W]", fontsize=16)
                axes[i].set_yscale('log')

            else:
                # Regular parameter plots with highlighted best solutions
                if not df_bo_best_iter_combined.empty:
                    axes[i].plot(df_bo_best_iter_combined['iteration'], df_bo_best_iter_combined[param],
                                 marker='o', linestyle='-', color='blue', label='BO', linewidth=2)

                    # Highlight best BO solution
                    if best_bo_idx is not None:
                        best_iter = best_bo_solution['iteration']
                        best_param_val = best_bo_solution[param] if param in best_bo_solution else best_bo_solution[
                                                                                                       'dZ'] - \
                                                                                                   best_bo_solution[
                                                                                                       'dK']
                        axes[i].scatter([best_iter], [best_param_val], color='red', s=120,
                                        marker='*', edgecolors='black', linewidth=2, zorder=5)

                if not df_nsga3_best_iter_combined.empty:
                    axes[i].plot(df_nsga3_best_iter_combined['iteration'], df_nsga3_best_iter_combined[param],
                                 marker='x', linestyle='--', color='orange', label='NSGA-III', linewidth=2)

                    # Highlight best NSGA-III solution
                    if best_nsga3_idx is not None:
                        best_iter = best_nsga3_solution['iteration']
                        best_param_val = best_nsga3_solution[param] if param in best_nsga3_solution else \
                        best_nsga3_solution['dZ'] - best_nsga3_solution['dK']
                        axes[i].scatter([best_iter], [best_param_val], color='darkred', s=120,
                                        marker='*', edgecolors='black', linewidth=2, zorder=5)

                # Set proper labels for parameters
                if param == "zeta":
                    y_label = "γ"
                elif param == "clearance":
                    y_label = "clearance [µm]"
                elif param == "LKG":
                    y_label = "LKG [mm]"
                elif param == "lF":
                    y_label = "lF [mm]"
                else:
                    y_label = param

                axes[i].set_ylabel(y_label, fontsize=16)

                # Calculate y-limits for parameters
                all_param_values = []
                if not df_bo_best_iter_combined.empty and param in df_bo_best_iter_combined.columns:
                    all_param_values.extend(df_bo_best_iter_combined[param].tolist())
                if not df_nsga3_best_iter_combined.empty and param in df_nsga3_best_iter_combined.columns:
                    all_param_values.extend(df_nsga3_best_iter_combined[param].tolist())

                if all_param_values:
                    min_val = min(all_param_values)
                    max_val = max(all_param_values)
                    if min_val == max_val:
                        pad = 0.1 * abs(min_val) if min_val != 0 else 0.1
                        axes[i].set_ylim(min_val - pad, max_val + pad)
                    else:
                        range_val = max_val - min_val
                        axes[i].set_ylim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)

            axes[i].tick_params(axis='both', labelsize=14)
            axes[i].grid(True, alpha=0.3)

            # Create custom legend with best solution indicators
            legend_elements = axes[i].get_legend_handles_labels()
            if i == 0:  # Add special legend elements only to the first plot
                from matplotlib.patches import Patch
                from matplotlib.lines import Line2D
                legend_elements[0].append(Line2D([0], [0], marker='*', color='red', linewidth=0,
                                                 markersize=12, label='Best Solutions', markeredgecolor='black'))
            axes[i].legend(handles=legend_elements[0], labels=legend_elements[1], fontsize=14)

        axes[-1].set_xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
        axes[-1].tick_params(axis='both', labelsize=14)

        fig.suptitle("Parameter Evolution with Best Solutions Highlighted", fontsize=18, y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / "enhanced_combined_param_evolution_with_best_solutions.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("Enhanced plots with best solution highlighting created successfully!")


def create_best_solution_summary_plot(df_bo, df_nsga3, output_dir):
    """
    Create a summary plot showing the best solutions and their characteristics.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    best_solutions = []

    # Find best solutions
    if not df_bo.empty:
        best_bo_idx = df_bo['total'].idxmin()
        best_bo = df_bo.loc[best_bo_idx]
        best_solutions.append({
            'optimizer': 'BO',
            'iteration': best_bo.get('iteration', 'N/A'),
            'total': best_bo['total'],
            'mechanical': best_bo.get('mechanical', 0),
            'volumetric': best_bo.get('volumetric', 0),
            'clearance': best_bo.get('clearance', best_bo['dZ'] - best_bo['dK']),
            'dK': best_bo['dK'],
            'dZ': best_bo['dZ'],
            'LKG': best_bo['LKG'],
            'lF': best_bo['lF'],
            'zeta': best_bo['zeta']
        })

    if not df_nsga3.empty:
        best_nsga3_idx = df_nsga3['total'].idxmin()
        best_nsga3 = df_nsga3.loc[best_nsga3_idx]
        best_solutions.append({
            'optimizer': 'NSGA-III',
            'iteration': best_nsga3.get('iteration', 'N/A'),
            'total': best_nsga3['total'],
            'mechanical': best_nsga3.get('mechanical', 0),
            'volumetric': best_nsga3.get('volumetric', 0),
            'clearance': best_nsga3.get('clearance', best_nsga3['dZ'] - best_nsga3['dK']),
            'dK': best_nsga3['dK'],
            'dZ': best_nsga3['dZ'],
            'LKG': best_nsga3['LKG'],
            'lF': best_nsga3['lF'],
            'zeta': best_nsga3['zeta']
        })

    if not best_solutions:
        return

    df_best = pd.DataFrame(best_solutions)

    # Plot 1: Power losses comparison
    losses = ['mechanical', 'volumetric', 'total']
    x_pos = np.arange(len(losses))
    width = 0.35

    for i, opt in enumerate(df_best['optimizer']):
        values = [df_best[df_best['optimizer'] == opt][loss].iloc[0] for loss in losses]
        iteration = df_best[df_best['optimizer'] == opt]['iteration'].iloc[0]
        color = 'blue' if opt == 'BO' else 'orange'
        ax1.bar(x_pos + i * width, values, width, label=f'{opt} (Iter/Gen {iteration})', color=color, alpha=0.7)

    ax1.set_xlabel('Loss Type')
    ax1.set_ylabel('Power Loss [W]')
    ax1.set_title('Best Solutions: Power Losses Comparison')
    ax1.set_yscale('log')
    ax1.set_xticks(x_pos + width / 2)
    ax1.set_xticklabels(['Mechanical', 'Volumetric', 'Total'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Geometric parameters comparison
    geo_params = ['dK', 'dZ', 'clearance', 'LKG', 'lF']
    x_pos = np.arange(len(geo_params))

    for i, opt in enumerate(df_best['optimizer']):
        values = [df_best[df_best['optimizer'] == opt][param].iloc[0] for param in geo_params]
        iteration = df_best[df_best['optimizer'] == opt]['iteration'].iloc[0]
        color = 'blue' if opt == 'BO' else 'orange'
        ax2.bar(x_pos + i * width, values, width, label=f'{opt} (Iter/Gen {iteration})', color=color, alpha=0.7)

    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('Value [mm]')
    ax2.set_title('Best Solutions: Geometric Parameters')
    ax2.set_xticks(x_pos + width / 2)
    ax2.set_xticklabels(['dK', 'dZ', 'clearance', 'LKG', 'lF'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Zeta parameter comparison
    zeta_values = [df_best[df_best['optimizer'] == opt]['zeta'].iloc[0] for opt in df_best['optimizer']]
    colors = ['blue' if opt == 'BO' else 'orange' for opt in df_best['optimizer']]
    labels = [f"{opt} (Iter/Gen {df_best[df_best['optimizer'] == opt]['iteration'].iloc[0]})"
              for opt in df_best['optimizer']]

    ax3.bar(range(len(zeta_values)), zeta_values, color=colors, alpha=0.7)
    ax3.set_xlabel('Optimizer')
    ax3.set_ylabel('γ (Zeta)')
    ax3.set_title('Best Solutions: Zeta Parameter')
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary table
    ax4.axis('tight')
    ax4.axis('off')

    # Create summary table data
    table_data = []
    for _, row in df_best.iterrows():
        table_data.append([
            row['optimizer'],
            f"Iter/Gen {row['iteration']}",
            f"{row['total']:.2e}",
            f"{row['mechanical']:.2e}",
            f"{row['volumetric']:.2e}",
            f"{row['clearance']:.3f}",
            f"{row['dK']:.3f}",
            f"{row['dZ']:.3f}",
            f"{row['LKG']:.1f}",
            f"{row['lF']:.1f}",
            f"{row['zeta']:.0f}"
        ])

    columns = ['Optimizer', 'Iter/Gen', 'Total Loss\n[W]', 'Mech Loss\n[W]', 'Vol Loss\n[W]',
               'Clearance\n[mm]', 'dK\n[mm]', 'dZ\n[mm]', 'LKG\n[mm]', 'lF\n[mm]', 'γ']

    table = ax4.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Color header and alternate rows
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax4.set_title('Best Solutions Summary', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "best_solutions_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Best solutions summary plot created successfully!")

    return df_best

def check_required_files(folder_path):
    """
    Check if all required files exist for contact pressure analysis.
    Returns both missing files and available alternatives.
    """
    # Primary files needed for full T1 analysis
    primary_files = [
        os.path.join(folder_path, 'input', 'geometry.txt'),
        os.path.join(folder_path, 'input', 'operatingconditions.txt'),
        os.path.join(folder_path, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt'),
        os.path.join(folder_path, 'output', 'piston', 'piston.txt')
    ]

    # Alternative files that might exist
    alternative_files = [
        os.path.join(folder_path, 'output', 'piston', 'piston.txt'),
        os.path.join(folder_path, 'output', 'piston.txt'),
        os.path.join(folder_path, 'output', 'results.txt'),
    ]

    missing_primary = []
    available_alternatives = []

    for file_path in primary_files:
        if not os.path.exists(file_path):
            missing_primary.append(file_path)

    for file_path in alternative_files:
        if os.path.exists(file_path):
            available_alternatives.append(file_path)

    return missing_primary, available_alternatives


def extract_contact_pressure_from_piston_file(piston_file_path):
    """
    Extract contact pressure information from piston.txt file if available.
    This is a fallback when the matlab contact pressure files don't exist.
    """
    try:
        import pandas as pd
        df = pd.read_csv(piston_file_path, delimiter='\t')

        print(f"    📋 File columns: {list(df.columns)}")

        # Look for contact pressure related columns with more specific patterns
        potential_cp_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['contact', 'pressure', 'force', 'load']):
                if not any(exclude in col_lower for exclude in ['loss', 'power', 'volume']):
                    potential_cp_cols.append(col)

        print(f"    🔍 Potential contact pressure columns: {potential_cp_cols}")

        if potential_cp_cols:
            # Use the revolution data (similar to T2 analysis)
            df_filtered = df[df['revolution'] <= 6.0] if 'revolution' in df.columns else df
            print(f"    📊 Data rows: {len(df)} total, {len(df_filtered)} filtered")

            contact_pressure_data = {}
            for col in potential_cp_cols:
                if df_filtered[col].notna().any():
                    col_mean = float(df_filtered[col].mean())
                    col_max = float(df_filtered[col].max())
                    col_min = float(df_filtered[col].min())

                    contact_pressure_data[col] = {
                        'mean': col_mean,
                        'max': col_max,
                        'min': col_min
                    }
                    print(f"    📈 {col}: mean={col_mean:.2e}, max={col_max:.2e}, min={col_min:.2e}")

            return contact_pressure_data
        else:
            print(f"    ❌ No contact pressure related columns found")
            return None

    except Exception as e:
        print(f"    ❌ Error reading piston file: {e}")
        return None


def check_file_structure(folder_path):
    """
    Check what files actually exist and their structure.
    """
    print(f"  🔍 Checking file structure for: {os.path.basename(folder_path)}")

    # Check all possible locations
    file_locations = [
        ('input/geometry.txt', os.path.join(folder_path, 'input', 'geometry.txt')),
        ('input/operatingconditions.txt', os.path.join(folder_path, 'input', 'operatingconditions.txt')),
        ('output/piston/matlab/Piston_Contact_Pressure.txt',
         os.path.join(folder_path, 'output', 'piston', 'matlab', 'Piston_Contact_Pressure.txt')),
        ('output/piston/piston.txt', os.path.join(folder_path, 'output', 'piston', 'piston.txt')),
        ('output/piston.txt', os.path.join(folder_path, 'output', 'piston.txt')),
    ]

    for name, path in file_locations:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(
            f"    {'✅' if exists else '❌'} {name}: {'exists' if exists else 'missing'} {f'({size} bytes)' if exists else ''}")

    # Check if matlab directory exists at all
    matlab_dir = os.path.join(folder_path, 'output', 'piston', 'matlab')
    if os.path.exists(matlab_dir):
        matlab_files = os.listdir(matlab_dir)
        print(f"    📁 Matlab directory contents: {matlab_files}")
    else:
        print(f"    ❌ Matlab directory doesn't exist: {matlab_dir}")


def run_contact_pressure_for_best_results(best_folders, output_dir):
    """
    Run contact pressure analysis for the best simulation folders using available data.
    Falls back to piston.txt analysis when full T1 analysis isn't possible.
    """
    print("\n" + "=" * 60)
    print("RUNNING CONTACT PRESSURE ANALYSIS FOR BEST RESULTS")
    print("=" * 60)

    contact_pressure_results = []
    skipped_count = 0
    fallback_count = 0

    # First, let's check a few folders to understand the structure
    print("\n🔍 DIAGNOSTIC: Checking file structure for first few folders...")
    for i, folder_info in enumerate(best_folders[:3]):
        folder_path = folder_info['folder_name']
        check_file_structure(folder_path)
        print()

    for i, folder_info in enumerate(best_folders):
        folder_path = folder_info['folder_name']
        optimizer = folder_info['optimizer']

        print(f"\n[{i + 1}/{len(best_folders)}] Processing {optimizer}: {os.path.basename(folder_path)}")

        # Check if required files exist
        missing_files, available_alternatives = check_required_files(folder_path)

        # Try full T1 analysis first
        can_do_full_analysis = len(missing_files) == 0

        if can_do_full_analysis:
            try:
                # Extract parameters from folder for T1 analysis
                geom_file = os.path.join(folder_path, 'input', 'geometry.txt')
                op_file = os.path.join(folder_path, 'input', 'operatingconditions.txt')

                lF_val = parse_geometry(geom_file)
                speed_val, hp_val = parse_operating_conditions(op_file)

                if lF_val is None or speed_val is None or hp_val is None:
                    print(f"  ❌ Could not parse geometry or operating conditions")
                    skipped_count += 1
                    continue

                print(f"  📊 Parameters: lF={lF_val:.1f}mm, speed={speed_val:.0f}rpm, ΔP={hp_val:.1f}")
                print(f"  🔄 Running full T1 contact pressure analysis...")

                # Run contact pressure analysis using T1 function
                result = create_cumulative_pressure_map(
                    filepath=folder_path,
                    m=50,
                    lF=lF_val,
                    n=speed_val,
                    deltap=hp_val,
                    plots=360,
                    degstep=1,
                    ignore=0,
                    offset=0,
                    minplot=0,
                    maxplot=0
                )

                if result is not None:
                    # Extract mean cumulative contact pressure
                    mean_cumulative_pressure = np.mean(result['cumulative'])
                    max_cumulative_pressure = np.max(result['cumulative'])

                    contact_pressure_results.append({
                        'folder_name': folder_path,
                        'optimizer': optimizer,
                        'analysis_type': 'Full T1 Analysis',
                        'max_cumulative_pressure': max_cumulative_pressure,
                        'mean_cumulative_pressure': mean_cumulative_pressure,
                        'mechanical': folder_info['mechanical'],
                        'volumetric': folder_info['volumetric'],
                        'total': folder_info['total'],
                        'dK': folder_info['dK'],
                        'dZ': folder_info['dZ'],
                        'LKG': folder_info['LKG'],
                        'lF': folder_info['lF'],
                        'zeta': folder_info['zeta'],
                        'clearance': folder_info['clearance'],
                        'result': result
                    })
                    print(
                        f"  ✅ Success - Mean: {mean_cumulative_pressure:.2e} Pa, Max: {max_cumulative_pressure:.2e} Pa")
                    continue
                else:
                    print(f"  ⚠️ T1 analysis returned None, trying fallback method...")

            except Exception as e:
                print(f"  ⚠️ T1 analysis failed: {str(e)[:80]}..., trying fallback method...")

        # Fallback: Try to extract contact pressure from piston.txt
        piston_files_to_try = [
            os.path.join(folder_path, 'output', 'piston', 'piston.txt'),
            os.path.join(folder_path, 'output', 'piston.txt')
        ]

        fallback_success = False
        for piston_file in piston_files_to_try:
            if os.path.exists(piston_file):
                print(f"  🔄 Attempting fallback analysis from: {os.path.relpath(piston_file, folder_path)}")

                contact_data = extract_contact_pressure_from_piston_file(piston_file)
                if contact_data:
                    # Choose the best contact pressure column (prefer "contact" in name)
                    best_col = None
                    for col in contact_data.keys():
                        if 'contact' in col.lower():
                            best_col = col
                            break
                    if not best_col:
                        best_col = list(contact_data.keys())[0]

                    cp_stats = contact_data[best_col]

                    contact_pressure_results.append({
                        'folder_name': folder_path,
                        'optimizer': optimizer,
                        'analysis_type': f'Fallback ({best_col})',
                        'max_cumulative_pressure': cp_stats['max'],
                        'mean_cumulative_pressure': cp_stats['mean'],
                        'mechanical': folder_info['mechanical'],
                        'volumetric': folder_info['volumetric'],
                        'total': folder_info['total'],
                        'dK': folder_info['dK'],
                        'dZ': folder_info['dZ'],
                        'LKG': folder_info['LKG'],
                        'lF': folder_info['lF'],
                        'zeta': folder_info['zeta'],
                        'clearance': folder_info['clearance'],
                        'result': None
                    })
                    print(f"  ✅ Fallback success - Mean: {cp_stats['mean']:.2e} Pa, Max: {cp_stats['max']:.2e} Pa")
                    fallback_count += 1
                    fallback_success = True
                    break
                else:
                    print(f"    ❌ No contact pressure columns found in {os.path.basename(piston_file)}")

        if not fallback_success:
            print(f"  ❌ No usable contact pressure data found")
            if missing_files:
                print(f"    Missing files: {len(missing_files)} files")
                for missing_file in missing_files[:2]:
                    print(f"    - {os.path.relpath(missing_file, folder_path)}")
            skipped_count += 1

    print(f"\n✅ Contact pressure analysis completed:")
    print(f"   - Full T1 analysis: {len(contact_pressure_results) - fallback_count} simulations")
    print(f"   - Fallback analysis: {fallback_count} simulations")
    print(f"   - Total successful: {len(contact_pressure_results)} simulations")
    print(f"   - Skipped: {skipped_count} simulations")
    print(f"   - Total processed: {len(best_folders)} simulations")

    return contact_pressure_results


def create_contact_pressure_plots(contact_results, output_dir):
    """
    Create plots showing contact pressure analysis results.
    """
    if not contact_results:
        print("❌ No contact pressure results to plot - all simulations were skipped due to missing files")
        print("\n💡 To fix this issue:")
        print("   1. Ensure the simulation data files exist in the expected locations")
        print("   2. Check that simulations have completed successfully")
        print("   3. Verify the file paths in the folder structure")
        return

    print(f"\n📊 Creating contact pressure plots for {len(contact_results)} successful simulations...")

    df_contact = pd.DataFrame(contact_results)

    # 1. Mean cumulative contact pressure comparison
    plt.figure(figsize=(12, 6))

    # Separate by optimizer
    bo_data = df_contact[df_contact['optimizer'] == 'BO']
    nsga3_data = df_contact[df_contact['optimizer'] == 'NSGA-III']

    # Create positions for bars
    x_positions = []
    labels = []
    colors = []
    pressures = []

    # Add BO data
    for i, (_, row) in enumerate(bo_data.iterrows()):
        x_positions.append(i)
        labels.append(f"BO-{i + 1}")
        colors.append('blue')
        pressures.append(row['max_cumulative_pressure'])

    # Add NSGA-III data
    nsga_start = len(bo_data)
    for i, (_, row) in enumerate(nsga3_data.iterrows()):
        x_positions.append(nsga_start + i)
        labels.append(f"NSGA-{i + 1}")
        colors.append('orange')
        pressures.append(row['max_cumulative_pressure'])

    if pressures:
        plt.bar(x_positions, pressures, color=colors, alpha=0.7, width=0.8)
        plt.xlabel('Simulation', fontsize=16)
        plt.ylabel('Max Cumulative Contact Pressure [Pa]', fontsize=16)
        plt.title(
            f'Max Cumulative Contact Pressure for Best Optimization Results\n({len(bo_data)} BO + {len(nsga3_data)} NSGA-III)',
            fontsize=14)
        plt.yscale('log')
        plt.xticks(x_positions, labels, rotation=45)

        # Add legend
        bo_patch = plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label=f'BO ({len(bo_data)})')
        nsga3_patch = plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.7, label=f'NSGA-III ({len(nsga3_data)})')
        plt.legend(handles=[bo_patch, nsga3_patch], fontsize=14)

        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "contact_pressure_comparison.png", dpi=300)
        plt.close()

    # Create additional plots for contact pressure vs losses
    for loss_type, loss_col in [('volumetric', 'volumetric'), ('mechanical', 'mechanical'), ('total', 'total')]:
        plt.figure(figsize=(8, 6))

        if not bo_data.empty:
            plt.scatter(bo_data[loss_col], bo_data['max_cumulative_pressure'],
                        color='blue', alpha=0.7, s=80, label=f'BO ({len(bo_data)})', marker='o')

        if not nsga3_data.empty:
            plt.scatter(nsga3_data[loss_col], nsga3_data['max_cumulative_pressure'],
                        color='orange', alpha=0.7, s=80, label=f'NSGA-III ({len(nsga3_data)})', marker='^')

        plt.xlabel(f'{loss_type.title()} Loss [W]', fontsize=16)
        plt.ylabel('Max Cumulative Contact Pressure [Pa]', fontsize=16)
        plt.title(f'Contact Pressure vs {loss_type.title()} Loss', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"contact_pressure_vs_{loss_type}_loss.png", dpi=300)
        plt.close()

    print(f"✅ Contact pressure plots saved to {output_dir}")


def main():
    """
    Updated main function with enhanced plotting that highlights best solutions.
    """

    # # Update these paths to your actual folders
    # bo_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run6_diameter_psiton_bushing_length_piston_advance_nsga\bayesian_optimization'
    # nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run6_diameter_psiton_bushing_length_piston_advance_nsga\advanced_nsga3'
    bo_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\bayesian_optimization'
    # nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\advanced_nsga3'
    nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run8_diameter_z_length_p_z\simple_nsga3'

    # Load results using updated parsing
    print("=" * 60)
    print("LOADING OPTIMIZATION RESULTS WITH NEW FOLDER FORMAT")
    print("=" * 60)
    print(f"BO folder: {bo_folder}")
    print(f"NSGA-III folder: {nsga3_folder}")

    bo_results = load_results(bo_folder, "BO")
    print(f"✅ Loaded {len(bo_results)} valid BO results")

    nsga3_results = load_results(nsga3_folder, "NSGA-III")
    print(f"✅ Loaded {len(nsga3_results)} valid NSGA-III results")

    # Convert to DataFrame for convenience
    df_bo = pd.DataFrame(bo_results)
    df_nsga3 = pd.DataFrame(nsga3_results)

    # Ensure a common iteration column name for both
    if 'generation' in df_nsga3.columns:
        df_nsga3 = df_nsga3.rename(columns={'generation': 'iteration'})

    # Create output directory for plots
    output_dir = Path("optimization_plots_enhanced_with_best_solutions")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nData Summary:")
    if not df_bo.empty:
        print(f"BO data shape: {df_bo.shape}")
        print(f"Sample BO parameters:")
        sample_cols = ['dK', 'dZ', 'clearance', 'LKG', 'lF', 'zeta', 'total']
        for col in sample_cols:
            if col in df_bo.columns:
                print(f"  {col}: {df_bo[col].iloc[0]:.6f}" if col != 'zeta' else f"  {col}: {df_bo[col].iloc[0]:.0f}")

    if not df_nsga3.empty:
        print(f"NSGA-III data shape: {df_nsga3.shape}")
        print(f"Sample NSGA-III parameters:")
        sample_cols = ['dK', 'dZ', 'clearance', 'LKG', 'lF', 'zeta', 'total']
        for col in sample_cols:
            if col in df_nsga3.columns:
                print(
                    f"  {col}: {df_nsga3[col].iloc[0]:.6f}" if col != 'zeta' else f"  {col}: {df_nsga3[col].iloc[0]:.0f}")

    # Define parameter names for plotting
    parameters = ["dK", "dZ", "clearance", "LKG", "lF", "zeta"]

    print("\n" + "=" * 60)
    print("CREATING ENHANCED PLOTS WITH BEST SOLUTION HIGHLIGHTING")
    print("=" * 60)

    # Create enhanced combined plots with best solution highlighting
    create_enhanced_combined_plots(df_bo, df_nsga3, output_dir)

    # Create best solution summary plot
    df_best_solutions = create_best_solution_summary_plot(df_bo, df_nsga3, output_dir)

    # Print best solution details
    if df_best_solutions is not None and not df_best_solutions.empty:
        print("\n" + "=" * 60)
        print("BEST SOLUTION DETAILS")
        print("=" * 60)

        for _, solution in df_best_solutions.iterrows():
            print(f"\n{solution['optimizer']} Best Solution (Iteration/Generation {solution['iteration']}):")
            print(f"  Total Loss: {solution['total']:.4e} W")
            print(f"  Mechanical Loss: {solution['mechanical']:.4e} W")
            print(f"  Volumetric Loss: {solution['volumetric']:.4e} W")
            print(f"  Parameters:")
            print(f"    dK: {solution['dK']:.4f} mm")
            print(f"    dZ: {solution['dZ']:.4f} mm")
            print(f"    Clearance: {solution['clearance']:.4f} mm")
            print(f"    LKG: {solution['LKG']:.1f} mm")
            print(f"    lF: {solution['lF']:.1f} mm")
            print(f"    γ (zeta): {solution['zeta']:.0f}")

    # --- CREATE ORIGINAL PLOTS (without best solution highlighting) ---
    print("\n" + "=" * 60)
    print("CREATING STANDARD OPTIMIZATION PLOTS")
    print("=" * 60)

    # 1. Individual Convergence Plots
    if not df_bo.empty:
        plt.figure(figsize=(8, 5))
        df_bo_best = df_bo.groupby('iteration')['total'].min().reset_index()
        plt.plot(df_bo_best['iteration'], df_bo_best['total'], marker='o', label='BO')
        plt.title("BO Convergence: Best Total Loss vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Best Total Loss (min up to iteration)")
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_dir / "convergence_BO.png")
        plt.close()

    if not df_nsga3.empty:
        plt.figure(figsize=(8, 5))
        df_nsga3_best = df_nsga3.groupby('iteration')['total'].min().reset_index()
        plt.plot(df_nsga3_best['iteration'], df_nsga3_best['total'], color='orange', marker='o', label='NSGA-III')
        plt.xlabel("Generation")
        plt.ylabel("Best Total Loss [W]")
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_dir / "convergence_NSGA3.png")
        plt.close()

    # 2. Parameter Distribution Histograms
    if not df_bo.empty:
        fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
        for i, param in enumerate(parameters):
            sns.histplot(df_bo, x=param, kde=True, ax=axes[i], color='blue', bins=20)
            axes[i].set_title(f"BO: Distribution of {param}")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_dir / "param_distribution_BO.png")
        plt.close(fig)

    if not df_nsga3.empty:
        fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
        for i, param in enumerate(parameters):
            sns.histplot(df_nsga3, x=param, kde=True, ax=axes[i], color='orange', bins=20)
            axes[i].set_title(f"NSGA-III: Distribution of {param}")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(output_dir / "param_distribution_NSGA3.png")
        plt.close(fig)

    # 3. Parameter vs Total Loss Scatter Plots
    if not df_bo.empty:
        fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
        for i, param in enumerate(parameters):
            axes[i].scatter(df_bo[param], df_bo["total"], color='blue', alpha=0.6)
            axes[i].set_title(f"BO: {param} vs Total Loss")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Total Loss")
            axes[i].set_yscale('log')
            axes[i].grid(True)
        fig.tight_layout()
        fig.savefig(output_dir / "param_vs_loss_BO.png")
        plt.close(fig)

    if not df_nsga3.empty:
        fig, axes = plt.subplots(len(parameters), 1, figsize=(8, 12))
        for i, param in enumerate(parameters):
            axes[i].scatter(df_nsga3[param], df_nsga3["total"], color='orange', alpha=0.6)
            axes[i].set_title(f"NSGA-III: {param} vs Total Loss")
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Total Loss")
            axes[i].set_yscale('log')
            axes[i].grid(True)
        fig.tight_layout()
        fig.savefig(output_dir / "param_vs_loss_NSGA3.png")
        plt.close(fig)

    # 4. Individual Pareto Front Plots
    if not df_bo.empty and "mechanical" in df_bo.columns and "volumetric" in df_bo.columns:
        plt.figure(figsize=(6, 5))
        plt.scatter(df_bo["mechanical"], df_bo["volumetric"], color='blue', alpha=0.7)
        plt.title("BO: Mechanical vs Volumetric Loss")
        plt.xlabel("Mechanical Loss")
        plt.ylabel("Volumetric Loss")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(output_dir / "pareto_BO.png")
        plt.close()

    if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
        plt.figure(figsize=(6, 5))
        plt.scatter(df_nsga3["mechanical"], df_nsga3["volumetric"], color='orange', alpha=0.7)
        plt.title("NSGA-III: Mechanical vs Volumetric Loss")
        plt.xlabel("Mechanical Loss")
        plt.ylabel("Volumetric Loss")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(output_dir / "pareto_NSGA3.png")
        plt.close()

    # 5. Sensitivity Analysis
    # Compute clearance for sensitivity analysis
    if not df_bo.empty:
        df_bo["clearance"] = df_bo["dZ"] - df_bo["dK"]
    if not df_nsga3.empty:
        df_nsga3["clearance"] = df_nsga3["dZ"] - df_nsga3["dK"]

    selected_params = ["clearance", "zeta", "lF", "LKG"]
    param_labels = {
        "clearance": "clearance",
        "zeta": "γ",
        "lF": "lF",
        "LKG": "LKG"
    }

    sensitivity_data = []

    for param in selected_params:
        bo_corr = np.nan
        nsga_corr = np.nan

        if not df_bo.empty and param in df_bo.columns:
            bo_corr, _ = pearsonr(df_bo[param], df_bo["total"])
            bo_corr = abs(bo_corr)

        if not df_nsga3.empty and param in df_nsga3.columns:
            nsga_corr, _ = pearsonr(df_nsga3[param], df_nsga3["total"])
            nsga_corr = abs(nsga_corr)

        sensitivity_data.append({
            "parameter": param_labels[param],
            "BO": bo_corr,
            "NSGA-III": nsga_corr
        })

    # Plotting sensitivity
    df_sensitivity = pd.DataFrame(sensitivity_data)
    x = np.arange(len(df_sensitivity["parameter"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, df_sensitivity["BO"], width, label='BO', color='blue')
    bars2 = ax.bar(x + width / 2, df_sensitivity["NSGA-III"], width, label='NSGA-III', color='orange')

    ax.set_xlabel("Parameter", fontsize=16)
    ax.set_ylabel("Sensitivity (|Pearson Correlation|)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sensitivity["parameter"], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=18)
    ax.grid(True, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_bar_chart_all_params.png")
    plt.close(fig)

    # --- GET TOP RESULTS FOR CONTACT PRESSURE ANALYSIS ---
    print("\n" + "=" * 60)
    print("SELECTING TOP RESULTS FOR CONTACT PRESSURE ANALYSIS")
    print("=" * 60)

    top_n = 10
    best_folders = []

    # Get top N BO results by total loss
    if not df_bo.empty:
        top_bo = df_bo.nsmallest(top_n, 'total').copy()
        for _, row in top_bo.iterrows():
            best_folders.append(row.to_dict())
        print(f"✅ Selected {len(top_bo)} best BO results")

    # Get top N NSGA-III results from Pareto front
    if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
        pareto_nsga3 = find_pareto_front(df_nsga3, ["mechanical", "volumetric"])
        if not pareto_nsga3.empty:
            top_nsga3 = pareto_nsga3.sort_values(by='total').head(top_n).copy()
            for _, row in top_nsga3.iterrows():
                best_folders.append(row.to_dict())
            print(f"✅ Selected {len(top_nsga3)} best NSGA-III results from Pareto front")

    print(f"Total simulations selected for contact pressure analysis: {len(best_folders)}")

    # --- RUN CONTACT PRESSURE ANALYSIS ---
    contact_results = run_contact_pressure_for_best_results(best_folders, output_dir)

    # --- CREATE CONTACT PRESSURE PLOTS ---
    create_contact_pressure_plots(contact_results, output_dir)

    # --- SAVE RESULTS TO CSV ---
    if contact_results:
        df_contact_results = pd.DataFrame(contact_results)
        csv_columns = ['optimizer', 'analysis_type', 'max_cumulative_pressure', 'mean_cumulative_pressure',
                       'total', 'mechanical', 'volumetric', 'dK', 'dZ', 'clearance', 'LKG', 'lF', 'zeta', 'folder_name']
        df_contact_results[csv_columns].to_csv(
            output_dir / "contact_pressure_results.csv", index=False
        )
        print(f"✅ Contact pressure results saved to {output_dir / 'contact_pressure_results.csv'}")

    # Save top results summary
    top_n_save = 5
    columns_to_save = ["optimizer", "iteration", "total", "mechanical", "volumetric", "dK", "dZ", "clearance", "LKG",
                       "lF", "zeta"]

    # Get top 5 results from each optimizer
    top_bo_save = pd.DataFrame()
    if not df_bo.empty:
        top_bo_save = df_bo.nsmallest(top_n_save, 'total').copy()
        top_bo_save["optimizer"] = "BO"
        if 'clearance' not in top_bo_save.columns:
            top_bo_save['clearance'] = top_bo_save['dZ'] - top_bo_save['dK']
        top_bo_save = top_bo_save[columns_to_save]

    top_nsga3_save = pd.DataFrame()
    if not df_nsga3.empty:
        top_nsga3_save = df_nsga3.nsmallest(top_n_save, 'total').copy()
        top_nsga3_save["optimizer"] = "NSGA-III"
        if 'clearance' not in top_nsga3_save.columns:
            top_nsga3_save['clearance'] = top_nsga3_save['dZ'] - top_nsga3_save['dK']
        top_nsga3_save = top_nsga3_save[columns_to_save]

    # Combine and save
    df_top_combined = pd.concat([top_bo_save, top_nsga3_save], ignore_index=True)
    df_top_combined.to_csv(output_dir / "top5_optimal_results.csv", index=False)

    print(f"✅ Saved top 5 parameter sets to: {output_dir / 'top5_optimal_results.csv'}")

    print("\n" + "=" * 60)
    print("INTEGRATED ANALYSIS COMPLETED")
    print("=" * 60)
    if contact_results:
        full_analysis_count = sum(1 for r in contact_results if 'Full T1' in r.get('analysis_type', ''))
        fallback_count = len(contact_results) - full_analysis_count
        print(f"✅ SUCCESS: Analyzed {len(contact_results)} simulations with contact pressure data")
        if full_analysis_count > 0:
            print(f"   - Full T1 analysis: {full_analysis_count} simulations")
        if fallback_count > 0:
            print(f"   - Fallback analysis: {fallback_count} simulations")
    else:
        print("⚠️  WARNING: No simulations had complete data for contact pressure analysis")

    print(f"📁 All results saved to: {output_dir}")
    print("✅ Enhanced plots with best solution highlighting created")
    print("✅ Updated to handle new folder format with clearance (CL) parameter")

    # Print summary of generated plots
    print("\n📊 Generated Plots Summary:")
    print("   Enhanced plots (with best solution highlighting):")
    print("   - enhanced_combined_convergence.png")
    print("   - enhanced_combined_pareto.png")
    print("   - enhanced_combined_param_evolution_with_best_solutions.png")
    print("   - best_solutions_summary.png")
    print("   Standard plots:")
    print("   - Individual convergence, parameter distributions, sensitivity analysis")
    print("   - Contact pressure analysis plots (if data available)")


if __name__ == "__main__":
    main()