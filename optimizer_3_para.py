import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def find_pareto_front(df: pd.DataFrame, objective_cols: list, minimize: bool = True) -> pd.DataFrame:
    """Identifies the Pareto front from a DataFrame of solutions."""
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


def create_enhanced_plots(df_bo, df_nsga3, output_dir):
    """Create enhanced plots showing gamma influence and optimal solutions."""

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Debug data
    print(f"\nData Debug:")
    print(f"BO data shape: {df_bo.shape if not df_bo.empty else 'Empty'}")
    print(f"NSGA3 data shape: {df_nsga3.shape if not df_nsga3.empty else 'Empty'}")
    if not df_bo.empty:
        print(f"BO columns: {list(df_bo.columns)}")
        print(f"BO zeta range: {df_bo['zeta'].min():.1f} - {df_bo['zeta'].max():.1f}")
    if not df_nsga3.empty:
        print(f"NSGA3 columns: {list(df_nsga3.columns)}")
        print(f"NSGA3 zeta range: {df_nsga3['zeta'].min():.1f} - {df_nsga3['zeta'].max():.1f}")

    # 1. Enhanced Pareto Front with Gamma (zeta) as dot size
    plt.figure(figsize=(12, 8))

    has_data = False

    if not df_bo.empty and "mechanical" in df_bo.columns and "volumetric" in df_bo.columns:
        # Normalize zeta for size (0-5 degrees -> 20-200 size)
        bo_sizes = 20 + (df_bo["zeta"] / 5.0) * 180
        plt.scatter(df_bo["mechanical"], df_bo["volumetric"],
                    s=bo_sizes, color='blue', alpha=0.6,
                    label='BO', edgecolors='darkblue', linewidth=0.5)
        has_data = True

    if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
        # Normalize zeta for size
        nsga3_sizes = 20 + (df_nsga3["zeta"] / 5.0) * 180
        plt.scatter(df_nsga3["mechanical"], df_nsga3["volumetric"],
                    s=nsga3_sizes, color='orange', alpha=0.6,
                    label='NSGA-III', marker='^', edgecolors='darkorange', linewidth=0.5)
        has_data = True

    # Add Pareto fronts
    if not df_bo.empty and "mechanical" in df_bo.columns and "volumetric" in df_bo.columns:
        pareto_bo = find_pareto_front(df_bo, ["mechanical", "volumetric"])
        if not pareto_bo.empty:
            pareto_bo_sorted = pareto_bo.sort_values('mechanical')
            plt.plot(pareto_bo_sorted["mechanical"], pareto_bo_sorted["volumetric"],
                     'b-', linewidth=2, alpha=0.8, label='BO Pareto Front')

    if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
        pareto_nsga3 = find_pareto_front(df_nsga3, ["mechanical", "volumetric"])
        if not pareto_nsga3.empty:
            pareto_nsga3_sorted = pareto_nsga3.sort_values('mechanical')
            plt.plot(pareto_nsga3_sorted["mechanical"], pareto_nsga3_sorted["volumetric"],
                     'orange', linewidth=2, linestyle='--', alpha=0.8, label='NSGA-III Pareto Front')

    plt.xlabel("Mechanical Loss [W]", fontsize=16)
    plt.ylabel("Volumetric Loss [W]", fontsize=16)
    plt.title("Pareto Front with Inclination Angle γ (Dot Size: Small=0°, Large=5°)", fontsize=14)

    if has_data:
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "enhanced_pareto_with_gamma.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Convergence plot with best gamma values
    plt.figure(figsize=(12, 8))

    has_convergence_data = False

    if not df_bo.empty and 'iteration' in df_bo.columns:
        df_bo_best = df_bo.loc[df_bo.groupby('iteration')['total'].idxmin()]
        plt.plot(df_bo_best['iteration'], df_bo_best['total'],
                 marker='o', linestyle='-', color='blue', linewidth=2, markersize=6, label='BO')

        # Add gamma values as text annotations (every few points)
        for i, (idx, row) in enumerate(df_bo_best.iterrows()):
            if i % max(1, len(df_bo_best) // 5) == 0:  # Show every 5th point
                plt.annotate(f'γ={row["zeta"]:.0f}°',
                             (row['iteration'], row['total']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, color='blue', alpha=0.8)
        has_convergence_data = True

    if not df_nsga3.empty and 'iteration' in df_nsga3.columns:
        df_nsga3_best = df_nsga3.loc[df_nsga3.groupby('iteration')['total'].idxmin()]
        plt.plot(df_nsga3_best['iteration'], df_nsga3_best['total'],
                 marker='x', linestyle='--', color='orange', linewidth=2, markersize=8, label='NSGA-III')

        # Add gamma values as text annotations
        for i, (idx, row) in enumerate(df_nsga3_best.iterrows()):
            if i % max(1, len(df_nsga3_best) // 5) == 0:
                plt.annotate(f'γ={row["zeta"]:.0f}°',
                             (row['iteration'], row['total']),
                             xytext=(5, -10), textcoords='offset points',
                             fontsize=8, color='orange', alpha=0.8)
        has_convergence_data = True

    plt.xlabel("Optimization Step (Iteration/Generation)", fontsize=16)
    plt.ylabel("Best Total Loss [W]", fontsize=16)
    plt.title("Convergence with Inclination Angle γ Values", fontsize=14)

    if has_convergence_data:
        plt.yscale('log')
        plt.legend(fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_with_gamma.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Gamma influence on total loss
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # BO - Gamma vs Total Loss
    if not df_bo.empty:
        scatter_bo = axes[0].scatter(df_bo["zeta"], df_bo["total"],
                                     c=df_bo["total"], cmap='viridis', alpha=0.7, s=60)
        axes[0].set_xlabel("Inclination Angle γ [degrees]", fontsize=14)
        axes[0].set_ylabel("Total Loss [W]", fontsize=14)
        axes[0].set_title("BO: Inclination Angle vs Total Loss", fontsize=14)
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter_bo, ax=axes[0], label='Total Loss [W]')
    else:
        axes[0].text(0.5, 0.5, 'No BO Data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("BO: No Data Available", fontsize=14)

    # NSGA-III - Gamma vs Total Loss
    if not df_nsga3.empty:
        scatter_nsga3 = axes[1].scatter(df_nsga3["zeta"], df_nsga3["total"],
                                        c=df_nsga3["total"], cmap='plasma', alpha=0.7, s=60)
        axes[1].set_xlabel("Inclination Angle γ [degrees]", fontsize=14)
        axes[1].set_ylabel("Total Loss [W]", fontsize=14)
        axes[1].set_title("NSGA-III: Inclination Angle vs Total Loss", fontsize=14)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter_nsga3, ax=axes[1], label='Total Loss [W]')
    else:
        axes[1].text(0.5, 0.5, 'No NSGA-III Data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("NSGA-III: No Data Available", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / "gamma_vs_total_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Print optimal solutions
    print("\n" + "=" * 60)
    print("OPTIMAL SOLUTIONS ANALYSIS")
    print("=" * 60)

    top_n = 5

    if not df_bo.empty:
        top_bo = df_bo.nsmallest(top_n, 'total')
        print(f"\nTop {min(top_n, len(top_bo))} BO Solutions:")
        print("-" * 40)
        for i, (_, row) in enumerate(top_bo.iterrows()):
            print(
                f"{i + 1}. Total Loss: {row['total']:.2f}W (Mech: {row['mechanical']:.2f}W, Vol: {row['volumetric']:.2f}W)")
            print(f"   γ = {row['zeta']:.0f}°, dK = {row['dK']:.3f}mm, dZ = {row['dZ']:.3f}mm")
            if 'LKG' in row and 'lF' in row:
                print(f"   LKG = {row['LKG']:.2f}mm, lF = {row['lF']:.2f}mm")
    else:
        print("\nNo BO data available")

    if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
        pareto_nsga3 = find_pareto_front(df_nsga3, ["mechanical", "volumetric"])
        if not pareto_nsga3.empty:
            top_pareto = pareto_nsga3.nsmallest(top_n, 'total')
            print(f"\nTop {min(top_n, len(top_pareto))} NSGA-III Pareto Optimal Solutions:")
            print("-" * 50)
            for i, (_, row) in enumerate(top_pareto.iterrows()):
                print(
                    f"{i + 1}. Total Loss: {row['total']:.2f}W (Mech: {row['mechanical']:.2f}W, Vol: {row['volumetric']:.2f}W)")
                print(f"   γ = {row['zeta']:.0f}°, dK = {row['dK']:.3f}mm, dZ = {row['dZ']:.3f}mm")
                if 'LKG' in row and 'lF' in row:
                    print(f"   LKG = {row['LKG']:.2f}mm, lF = {row['lF']:.2f}mm")
        else:
            print("\nNo Pareto optimal solutions found for NSGA-III")
    else:
        print("\nNo NSGA-III data available or missing required columns")

    # 5. Summary statistics
    print(f"\n" + "=" * 60)
    print("GAMMA (INCLINATION ANGLE) STATISTICS")
    print("=" * 60)

    if not df_bo.empty and 'zeta' in df_bo.columns:
        print(f"\nBO Gamma Statistics:")
        print(f"  Range: {df_bo['zeta'].min():.1f}° - {df_bo['zeta'].max():.1f}°")
        print(f"  Mean: {df_bo['zeta'].mean():.2f}°")
        print(f"  Best solution gamma: {df_bo.loc[df_bo['total'].idxmin(), 'zeta']:.1f}°")
    else:
        print("\nNo BO gamma data available")

    if not df_nsga3.empty and 'zeta' in df_nsga3.columns:
        print(f"\nNSGA-III Gamma Statistics:")
        print(f"  Range: {df_nsga3['zeta'].min():.1f}° - {df_nsga3['zeta'].max():.1f}°")
        print(f"  Mean: {df_nsga3['zeta'].mean():.2f}°")
        print(f"  Best solution gamma: {df_nsga3.loc[df_nsga3['total'].idxmin(), 'zeta']:.1f}°")
        if not df_nsga3.empty and "mechanical" in df_nsga3.columns and "volumetric" in df_nsga3.columns:
            pareto_nsga3 = find_pareto_front(df_nsga3, ["mechanical", "volumetric"])
            if not pareto_nsga3.empty:
                best_pareto = pareto_nsga3.loc[pareto_nsga3['total'].idxmin()]
                print(f"  Best Pareto solution gamma: {best_pareto['zeta']:.1f}°")
    else:
        print("\nNo NSGA-III gamma data available")


def main():
    # Use raw strings or forward slashes for Windows paths
    bo_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run3_diameter_length_zeta_optimization_BO_more_generation\bayesian_optimization'
    nsga3_folder = r'Z:\Studenten\Mit\Inline_Thesis-Simulation\HSP\RUN\Run_optimizer_test\Run5_diameter_zeta_simple_nsga-III\simple_nsga3'

    # Import and use the load_results function from your main script
    # You need to have these functions available in your environment
    from plots_optimization.chat import load_results  # Replace with your actual module name

    bo_results = load_results(bo_folder, "BO")
    nsga3_results = load_results(nsga3_folder, "NSGA-III")

    # Convert to DataFrame
    df_bo = pd.DataFrame(bo_results)
    df_nsga3 = pd.DataFrame(nsga3_results)

    # Ensure iteration column exists for both
    if not df_nsga3.empty and 'generation' in df_nsga3.columns:
        df_nsga3 = df_nsga3.rename(columns={'generation': 'iteration'})

    # Create output directory
    output_dir = Path("enhanced_optimization_analysis")

    # Generate enhanced plots
    create_enhanced_plots(df_bo, df_nsga3, output_dir)

    print(f"\nEnhanced plots saved to: {output_dir}")
    print("Generated plots:")
    print("  1. enhanced_pareto_with_gamma.png - Shows gamma influence via dot size")
    print("  2. convergence_with_gamma.png - Convergence with gamma annotations")
    print("  3. gamma_vs_total_loss.png - Direct gamma vs loss relationship")



if __name__ == "__main__":
    main()