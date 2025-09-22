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

        # Original acceleration expression
        self.acc_expr = -R * (cos(theta) * tan(beta) / ((cos(theta) * tan(beta) * tan(phi) + 1) * cos(phi)) +
                              (1 - cos(theta)) * cos(theta) * tan(beta) ** 2 * tan(phi) /
                              ((cos(theta) * tan(beta) * tan(phi) + 1) ** 2 * cos(phi)) +
                              2 * ((1 - cos(theta)) * sin(theta) ** 2 * tan(beta) ** 3 * tan(phi) ** 2 /
                                   ((cos(theta) * tan(beta) * tan(phi) + 1) ** 3 * cos(phi)) +
                                   sin(theta) ** 2 * tan(beta) ** 2 * tan(phi) /
                                   ((cos(theta) * tan(beta) * tan(phi) + 1) ** 2 * cos(phi))))

        # New acceleration expression from the image
        K = tan(beta) * tan(phi)
        self.acc_expr_2 = -R * tan(beta) / cos(phi) * (K * (1 + sin(theta) ** 2) + cos(theta)) / (
                    1 + K * cos(theta)) ** 3

        subs = {R: self.params.R, beta: self.beta_rad, phi: self.phi_rad}
        self.disp_func = lambdify(theta, self.disp_expr.subs(subs))
        self.vel_func = lambdify(theta, self.vel_expr.subs(subs))
        self.acc_func = lambdify(theta, self.acc_expr.subs(subs))
        self.acc_func_2 = lambdify(theta, self.acc_expr_2.subs(subs))

    def calculate(self, theta_deg: float) -> Tuple[float, float, float, float]:
        theta_rad = np.deg2rad(theta_deg)
        disp = float(self.disp_func(theta_rad))
        vel = float(self.vel_func(theta_rad)) * self.params.omega
        acc1 = float(self.acc_func(theta_rad)) * self.params.omega ** 2
        acc2 = float(self.acc_func_2(theta_rad)) * self.params.omega ** 2
        return disp, vel, acc1, acc2

    def analyze_range(self, theta_deg_range: np.ndarray, save_path: Optional[str] = None) -> pd.DataFrame:
        results = np.array([self.calculate(t) for t in theta_deg_range])
        revolutions = theta_deg_range / 360
        df = pd.DataFrame({
            'theta': theta_deg_range,
            'revolution': revolutions,
            'displacement': results[:, 0],
            'velocity': results[:, 1],
            'acceleration_method1': results[:, 2],
            'acceleration_method2': results[:, 3]
        })

        if save_path:
            df.to_csv(save_path, index=False)

        return df

    @staticmethod
    def plot_results(df: pd.DataFrame, title: str = "Motion Analysis"):
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(4, 1, figure=fig, hspace=0.4)

        # Displacement plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['revolution'], df['displacement'], 'b-', linewidth=1.5)
        ax1.set_ylabel('Displacement (mm)')
        ax1.set_title('Displacement')
        ax1.grid(True, alpha=0.3)

        # Velocity plot
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df['revolution'], df['velocity'], 'g-', linewidth=1.5)
        ax2.set_ylabel('Velocity (mm/s)')
        ax2.set_title('Velocity')
        ax2.grid(True, alpha=0.3)

        # Both acceleration methods
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(df['revolution'], df['acceleration_method1'], 'r-', label='Method 1', linewidth=1.5)
        ax3.plot(df['revolution'], df['acceleration_method2'], 'b--', label='Method 2', linewidth=1.5)
        ax3.set_ylabel('Acceleration (mm/s²)')
        ax3.set_title('Acceleration Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Acceleration difference
        ax4 = fig.add_subplot(gs[3])
        diff = df['acceleration_method1'] - df['acceleration_method2']
        ax4.plot(df['revolution'], diff, 'm-', linewidth=1.5)
        ax4.set_xlabel('Revolution')
        ax4.set_ylabel('Acceleration Difference (mm/s²)')
        ax4.set_title('Acceleration Method Difference (Method 1 - Method 2)')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(title, y=1.02, fontsize=14)
        return fig


def main():
    # Example parameters
    params = MotionParams(R=43.5, beta=15, phi=5, omega=2 * np.pi * 2000 / 60, bore_length=33.25)

    # Analyze for 2 revolutions
    revolutions = 2
    theta_deg = np.linspace(0, 360 * revolutions, 360 * revolutions)

    # Create analyzer and get results
    analyzer = MotionAnalysis(params)
    results = analyzer.analyze_range(theta_deg, 'motion_results.csv')

    # Plot results
    fig = MotionAnalysis.plot_results(results, "Motion Analysis with Two Acceleration Methods")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
