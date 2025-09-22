import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, tan, cos, sin, lambdify
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class MotionParams:
    R: float
    beta: float
    phi: float
    omega: float
    bore_length: float


class MotionAnalysis:
    def __init__(self, params: MotionParams):
        self.params = params
        self.beta_rad = np.deg2rad(params.beta)
        self.phi_rad = np.deg2rad(params.phi)
        self._init_symbolic()

    def _init_symbolic(self):
        theta = self.theta = symbols('theta')
        R, beta, phi = symbols('R beta phi')

        self.disp_expr = -R * tan(beta) * (1 - cos(theta)) / (cos(phi) * (1 + tan(beta) * tan(phi) * cos(theta)))
        self.vel_expr = -R * ((1 - cos(theta)) * sin(theta) * tan(beta) ** 2 * tan(phi) /
                              ((cos(theta) * tan(beta) * tan(phi) + 1) ** 2 * cos(phi)) -
                              sin(theta) * tan(beta) / ((cos(theta) * tan(beta) * tan(phi) + 1) * cos(phi)))
        self.acc_expr = -R * (cos(theta) * tan(beta) / ((cos(theta) * tan(beta) * tan(phi) + 1) * cos(phi)) +
                              (1 - cos(theta)) * cos(theta) * tan(beta) ** 2 * tan(phi) /
                              ((cos(theta) * tan(beta) * tan(phi) + 1) ** 2 * cos(phi)) +
                              2 * ((1 - cos(theta)) * sin(theta) ** 2 * tan(beta) ** 3 * tan(phi) ** 2 /
                                   ((cos(theta) * tan(beta) * tan(phi) + 1) ** 3 * cos(phi)) +
                                   sin(theta) ** 2 * tan(beta) ** 2 * tan(phi) /
                                   ((cos(theta) * tan(beta) * tan(phi) + 1) ** 2 * cos(phi))))

        subs = {R: self.params.R, beta: self.beta_rad, phi: self.phi_rad}
        self.disp_func = lambdify(theta, self.disp_expr.subs(subs))
        self.vel_func = lambdify(theta, self.vel_expr.subs(subs))
        self.acc_func = lambdify(theta, self.acc_expr.subs(subs))

    def calculate(self, theta_deg: float) -> Tuple[float, float, float]:
        theta_rad = np.deg2rad(theta_deg)
        disp = float(self.disp_func(theta_rad))
        vel = float(self.vel_func(theta_rad)) * self.params.omega
        acc = float(self.acc_func(theta_rad)) * self.params.omega ** 2
        return disp, vel, acc

    def analyze_range(self, theta_deg_range: np.ndarray, save_path: Optional[str] = None) -> pd.DataFrame:
        results = np.array([self.calculate(t) for t in theta_deg_range])
        revolutions = theta_deg_range / 360
        df = pd.DataFrame({
            'theta': theta_deg_range,
            'revolution': revolutions,
            'displacement': results[:, 0],
            'velocity': results[:, 1],
            'acceleration': results[:, 2]
        })

        if save_path:
            df.to_csv(save_path, index=False)

        return df

    @staticmethod
    def plot_comparison(df1: pd.DataFrame, df2: pd.DataFrame, title: str = "Motion Comparison", phi1: float = 0,
                        phi2: float = 0):
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

        metrics = ['displacement', 'velocity', 'acceleration']
        ylabels = ['Displacement (mm)', 'Velocity (mm/s)', 'Acceleration (mm/s²)']

        for i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(df1['revolution'], df1[metric], 'b-', label=f'φ={phi1}°', linewidth=1.5)
            ax1.plot(df2['revolution'], df2[metric], 'r--', label=f'φ={phi2}°', linewidth=1.5)
            ax1.set_xlabel('Revolution')
            ax1.set_ylabel(ylabel)
            ax1.set_title(f'{metric.capitalize()} Comparison')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_xlim(0, max(df1['revolution'].max(), df2['revolution'].max()))

            ax2 = fig.add_subplot(gs[i, 1])
            diff = df1[metric] - df2[metric]
            ax2.plot(df1['revolution'], diff, 'g-', linewidth=1.5)
            ax2.set_xlabel('Revolution')
            ax2.set_ylabel(f'Δ {ylabel}')
            ax2.set_title(f'{metric.capitalize()} Difference')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, max(df1['revolution'].max(), df2['revolution'].max()))

        plt.suptitle(title, y=1.02, fontsize=14)
        return fig

class RadiusVariationAnalysis:
    def __init__(self, r_min: float, r_max: float, angle: float = 5, steps: int = 10):
        self.r_min = r_min
        self.r_max = r_max
        self.angle = angle
        self.steps = steps
        self.radius_values = np.linspace(r_min, r_max, steps)

    def analyze_radius_variation(self, beta: float, phi: float, omega: float, bore_length: float, revolutions: int = 1):
        results = []
        theta_deg = np.linspace(0, 360 * revolutions, 360 * revolutions)

        for radius in self.radius_values:
            params = MotionParams(R=radius, beta=beta, phi=phi, omega=omega, bore_length=bore_length)
            analyzer = MotionAnalysis(params)
            df = analyzer.analyze_range(theta_deg)
            results.append(df)

        return results

    def plot_radius_variation(self, results: list, title: str = "Radius Variation Analysis"):
        fig = plt.figure(figsize=(5,5))
        gs = plt.GridSpec(3, 1, figure=fig, hspace=0.4)

        metrics = [ 'acceleration']
        ylabels = [ 'Acceleration (mm/s²)']

        for i, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = fig.add_subplot(gs[i])

            for j, df in enumerate(results):
                radius = self.radius_values[j]
                ax.plot(df['revolution'], df[metric], label=f'R={radius:.1f}mm', linewidth=1.5)

            ax.set_xlabel('Revolution')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{metric.capitalize()} vs Revolution for Different Radii')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xlim(0, max(df['revolution'].max() for df in results))

        plt.suptitle(title, y=1.02, fontsize=14)
        return fig


def main():
    # Original analysis
    phi1, phi2 = 0, -10
    params1 = MotionParams(R=43.5, beta=15, phi=phi1, omega=2 * np.pi * 2000 / 60, bore_length=33.25)
    params2 = MotionParams(R=43.5, beta=15, phi=phi2, omega=2 * np.pi * 2000 / 60, bore_length=33.25)

    revolutions = 1
    theta_deg = np.linspace(0, 360 * revolutions, 360 * revolutions)

    analyzer1 = MotionAnalysis(params1)
    analyzer2 = MotionAnalysis(params2)

    results1 = analyzer1.analyze_range(theta_deg, 'results_phi1.csv')
    results2 = analyzer2.analyze_range(theta_deg, 'results_phi2.csv')

    fig1 = MotionAnalysis.plot_comparison(results1, results2, f"Comparison: φ={phi1}° vs φ={phi2}°", phi1, phi2)

    # Radius variation analysis
    r_min = 40
    r_max = r_min + 5  # 5mm increase as per angle requirement
    radius_analyzer = RadiusVariationAnalysis(r_min, r_max, angle=5, steps=10)
    radius_results = radius_analyzer.analyze_radius_variation(
        beta=15, phi=phi1, omega=2 * np.pi * 2000 / 60, bore_length=33.25, revolutions=revolutions
    )

    fig2 = radius_analyzer.plot_radius_variation(radius_results, "Motion Analysis for Different Radii")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()