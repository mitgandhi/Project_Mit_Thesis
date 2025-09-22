import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime


class SwashplatePiston:
    def __init__(self, swash_angle, piston_angle, radius_mm, pressure_mpa, piston_diameter_mm, rpm):
        """
        Initialize swashplate-piston calculator with metric units

        Args:
            swash_angle (float): Swashplate angle [degrees]
            piston_angle (float): Piston inclination angle [degrees]
            radius_mm (float): pitch radius [mm]
            pressure_mpa (float): Operating pressure [MPa]
            piston_diameter_mm (float): Piston diameter [mm]
            rpm (float): Rotational speed [RPM]
        """
        # Convert inputs to SI units for calculations
        self.alpha = np.radians(swash_angle)
        self.phi = np.radians(piston_angle)
        self.r = radius_mm / 1000  # mm to m
        self.P = pressure_mpa * 1e6  # MPa to Pa
        self.A = np.pi * (piston_diameter_mm / 2000) ** 2  # mm to m for area calculation
        self.omega = 2 * np.pi * rpm / 60  # RPM to rad/s

        # Store parameters for reference
        self.params = {
            'swash_angle': swash_angle,
            'piston_angle': piston_angle,
            'radius_mm': radius_mm,
            'pressure_mpa': pressure_mpa,
            'piston_diameter_mm': piston_diameter_mm,
            'rpm': rpm
        }

    def calculate_forces(self, theta):
        """Calculate forces at given rotation angle"""
        # Basic forces
        Fp = (self.P * self.A) * np.cos(theta)  # N
        Fa = Fp / (np.cos(self.alpha + self.phi))  # N
        Ft = Fa * np.sin(self.alpha + np.cos(self.phi))  # N
        Fs = Fp * np.tan(self.phi)  # N
        T = Fp * self.r * np.tan(self.alpha) / np.cos(self.phi)  # N·m

        # Constants for kinematics calculations
        K1 = (self.r * (np.tan(self.phi) * np.tan(self.alpha))) / (
                np.sin(self.phi) * (1 + np.tan(self.phi) * np.tan(self.alpha)))
        K2 = np.tan(self.phi) * np.tan(self.alpha)

        # Kinematics calculations
        z = (2 * K1) / (1 - K2) * np.cos(theta) * 1000  # Convert to mm
        v = K1 * self.omega * (1 + K2) * np.sin(theta) / (1 + K2 * np.cos(theta))  # m/s
        a = (self.omega ** 2 * K1 * (1 + K2) *
             (K2 * (1 + np.sin(theta) * np.sin(theta)) + np.cos(theta)) /
             ((1 + K2 * np.cos(theta)) ** 2))  # m/s²

        return {
            'piston_force_kn': Fp / 1000,  # Convert to kN
            'axial_force_kn': Fa / 1000,  # Convert to kN
            'tangential_force_kn': Ft / 1000,  # Convert to kN
            'side_force_kn': Fs / 1000,  # Convert to kN
            'drive_torque_nm': T,  # N·m
            'displacement_mm': z,  # mm
            'velocity_ms': v,  # m/s
            'acceleration_ms2': a  # m/s²
        }

    def plot_force_analysis(self, revolutions=1):
        """Create plots for forces with metric units"""
        n_points = 360 * revolutions
        theta = np.linspace(0, 2 * np.pi * revolutions, n_points)
        results = [self.calculate_forces(t) for t in theta]

        fig = plt.figure(figsize=(12, 20))
        plots = [
            ('piston_force_kn', 'Piston Force [kN]'),
            ('axial_force_kn', 'Axial Force [kN]'),
            ('tangential_force_kn', 'Tangential Force [kN]'),
            ('side_force_kn', 'Side Force [kN]'),
            ('drive_torque_nm', 'Drive Torque [N·m]'),
            ('displacement_mm', 'Displacement [mm]'),
            ('velocity_ms', 'Velocity [m/s]'),
            ('acceleration_ms2', 'Acceleration [m/s²]')
        ]

        for i, (key, ylabel) in enumerate(plots, 1):
            ax = fig.add_subplot(len(plots), 1, i)
            values = [r[key] for r in results]
            ax.plot(np.degrees(theta), values, 'b-', linewidth=2)
            ax.grid(True)
            ax.set_xlabel('Rotation Angle [degrees]')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs Rotation Angle')

        plt.tight_layout()
        return fig

    def create_3d_visualization(self):
        """Create animated 3D visualization of piston motion with metric units"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Set up trajectory
        theta = np.linspace(0, 2 * np.pi, 360)
        z = np.array([self.calculate_forces(t)['displacement_mm'] for t in theta])
        x = self.r * 1000 * np.cos(theta)  # Convert to mm
        y = self.r * 1000 * np.sin(theta)  # Convert to mm

        # Plot complete trajectory path
        ax.plot(x, y, z, 'b-', alpha=0.3, label='Piston Path')

        # Create swashplate surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, self.r * 1000, 20)  # Convert to mm
        U, V = np.meshgrid(u, v)
        X = V * np.cos(U)
        Y = V * np.sin(U)
        Z = V * np.tan(self.alpha) * np.cos(U)

        # Plot swashplate
        ax.plot_surface(X, Y, Z, alpha=0.4, color='gray')

        # Create piston position marker (updated in animation)
        point = ax.scatter([], [], [], color='red', s=100)

        # Create piston rod (line connecting piston to center)
        line, = ax.plot([], [], [], color='red', linewidth=2)

        # Set labels and title
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_title('Inclined Piston Motion on Swashplate')

        # Set axis limits
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Animation frame creation
        def update(frame):
            # Update piston position
            frame = frame % len(theta)  # Ensure continuous looping
            point._offsets3d = ([x[frame]], [y[frame]], [z[frame]])

            # Update connecting rod
            line.set_data_3d([0, x[frame]], [0, y[frame]], [0, z[frame]])

            # Rotate view for better visualization
            ax.view_init(elev=20, azim=frame)

            return point, line

        # Create animation
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, update, frames=len(theta),
                             interval=50, blit=True)

        return fig, anim


class SwashplateAnalyzer(SwashplatePiston):
    """Extended SwashplatePiston class with additional analysis capabilities"""

    def calculate_efficiency(self):
        """Calculate system efficiencies and power metrics"""
        # Mechanical efficiency (simplified model)
        friction_coefficient = 0.1
        mechanical_loss = friction_coefficient * (np.tan(self.alpha) + np.tan(self.phi))
        mechanical_efficiency = 1 - mechanical_loss

        # Volumetric efficiency (simplified model)
        leakage_factor = 0.05
        volumetric_efficiency = 1 - leakage_factor * (self.P / 1e6) / 100

        # Total efficiency
        total_efficiency = mechanical_efficiency * volumetric_efficiency

        # Power calculations
        theoretical_power = self.P * self.A * self.r * np.tan(self.alpha) * self.omega
        actual_power = theoretical_power * total_efficiency

        return {
            'mechanical_efficiency': mechanical_efficiency,
            'volumetric_efficiency': volumetric_efficiency,
            'total_efficiency': total_efficiency,
            'theoretical_power_kw': theoretical_power / 1000,  # Convert to kW
            'actual_power_kw': actual_power / 1000  # Convert to kW
        }

    def calculate_stresses(self):
        """Calculate critical component stresses in MPa"""
        # Bearing stress (simplified)
        bearing_load = self.P * self.A / np.cos(self.alpha)
        bearing_area = np.pi * self.r * 0.1  # Assumed bearing width
        bearing_stress = bearing_load / bearing_area

        # Piston stress
        piston_stress = self.P

        # Swashplate surface stress
        surface_stress = self.P * self.A * np.tan(self.alpha) / (np.pi * self.r ** 2)

        return {
            'bearing_stress_mpa': bearing_stress / 1e6,  # Convert to MPa
            'piston_stress_mpa': piston_stress / 1e6,  # Convert to MPa
            'surface_stress_mpa': surface_stress / 1e6  # Convert to MPa
        }


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime


# [Previous SwashplatePiston and SwashplateAnalyzer classes remain the same]

def main():
    st.set_page_config(page_title="Swashplate Analyzer", layout="wide")
    st.title("Swashplate Analysis System")

    # Helper function for numeric inputs
    def numeric_input_with_unit(label, min_val, max_val, default_val, unit, key=None):
        col1, col2 = st.sidebar.columns([7, 3])
        with col1:
            value = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=key
            )
        with col2:
            st.text(unit)
        return value

    # Sidebar for parameters
    st.sidebar.header("Parameters")

    try:
        # Parameter inputs with manual entry
        swash_angle = numeric_input_with_unit(
            "Swashplate Angle",
            min_val=0.0,
            max_val=30.0,
            default_val=15.0,
            unit="degrees",
            key="swash_angle"
        )

        piston_angle = numeric_input_with_unit(
            "Piston Angle",
            min_val=0.0,
            max_val=30.0,
            default_val=10.0,
            unit="degrees",
            key="piston_angle"
        )

        radius_mm = numeric_input_with_unit(
            "Radius",
            min_val=15.0,
            max_val=100.0,
            default_val=50.0,
            unit="mm",
            key="radius"
        )

        pressure_mpa = numeric_input_with_unit(
            "Pressure",
            min_val=1.0,
            max_val=40.0,
            default_val=20.0,
            unit="MPa",
            key="pressure"
        )

        piston_diameter_mm = numeric_input_with_unit(
            "Piston Diameter",
            min_val=10.0,
            max_val=50.0,
            default_val=35.0,
            unit="mm",
            key="piston_diameter"
        )

        rpm = numeric_input_with_unit(
            "Speed",
            min_val=0,
            max_val=6000,
            default_val=1500,
            unit="RPM",
            key="rpm"
        )

        # Add valid ranges reference
        st.sidebar.markdown("""
        ---
        **Valid Ranges:**
        - Swashplate Angle: 0-30°
        - Piston Angle: 0-30°
        - Radius: 15-100 mm
        - Pressure: 1-40 MPa
        - Piston Diameter: 10-50 mm
        - Speed: 0-6000 RPM
        """)

    except Exception as e:
        st.sidebar.error(f"Please enter valid numeric values within the specified ranges.")
        return

    # Presets with metric units
    st.sidebar.header("Presets")
    presets = {
        "High Speed": {
            'swash_angle': 15,
            'piston_angle': 5,
            'radius_mm': 78.37,
            'pressure_mpa': 250,
            'piston_diameter_mm': 21.2,
            'rpm': 6000
        },
        "High Pressure": {
            'swash_angle': 15,
            'piston_angle': 8,
            'radius_mm': 60,
            'pressure_mpa': 35,
            'piston_diameter_mm': 39.1,
            'rpm': 1200
        },
        "Balanced": {
            'swash_angle': 12,
            'piston_angle': 7,
            'radius_mm': 50,
            'pressure_mpa': 20,
            'piston_diameter_mm': 35.7,
            'rpm': 3000
        }
    }

    selected_preset = st.sidebar.selectbox("Load Preset", ["Custom"] + list(presets.keys()))

    # Create analyzer instance
    if selected_preset != "Custom":
        params = presets[selected_preset]
    else:
        params = {
            'swash_angle': swash_angle,
            'piston_angle': piston_angle,
            'radius_mm': radius_mm,
            'pressure_mpa': pressure_mpa,
            'piston_diameter_mm': piston_diameter_mm,
            'rpm': rpm
        }

    analyzer = SwashplateAnalyzer(**params)

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Analysis Results", "Force Analysis", "3D Visualization"])

    with tab1:
        efficiency_results = analyzer.calculate_efficiency()
        stress_results = analyzer.calculate_stresses()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Efficiency Analysis")
            st.write(f"Mechanical Efficiency: {efficiency_results['mechanical_efficiency']:.1%}")
            st.write(f"Volumetric Efficiency: {efficiency_results['volumetric_efficiency']:.1%}")
            st.write(f"Total Efficiency: {efficiency_results['total_efficiency']:.1%}")
            st.write(f"Theoretical Power: {efficiency_results['theoretical_power_kw']:.1f} kW")
            st.write(f"Actual Power: {efficiency_results['actual_power_kw']:.1f} kW")

        with col2:
            st.subheader("Stress Analysis")
            st.write(f"Bearing Stress: {stress_results['bearing_stress_mpa']:.1f} MPa")
            st.write(f"Piston Stress: {stress_results['piston_stress_mpa']:.1f} MPa")
            st.write(f"Surface Stress: {stress_results['surface_stress_mpa']:.1f} MPa")

    with tab2:
        st.pyplot(analyzer.plot_force_analysis())

    with tab3:
        st.write("3D Visualization of Piston Motion")

        # Add controls for visualization
        col1, col2 = st.columns([3, 1])
        with col2:
            st.write("Visualization Controls")
            view_static = st.checkbox("View Static Plot", value=False)
            if not view_static:
                animation_speed = st.slider("Animation Speed",
                                            min_value=1,
                                            max_value=100,
                                            value=50,
                                            help="Adjust animation speed")

        with col1:
            if view_static:
                # Show static plot
                fig = analyzer.create_3d_visualization()[0]
                st.pyplot(fig)
            else:
                # Show animated plot
                fig, anim = analyzer.create_3d_visualization()
                st.pyplot(fig)

                # Note about animation
                st.info("The animation shows the piston motion and swashplate interaction. "
                        "The red dot represents the piston position, and the red line shows "
                        "the connecting rod. The view rotates for better visualization.")

    # Export functionality
    if st.sidebar.button("Export Results"):
        theta = np.linspace(0, 2 * np.pi, 360)
        data = []
        for t in theta:
            forces = analyzer.calculate_forces(t)
            efficiency = analyzer.calculate_efficiency()
            stresses = analyzer.calculate_stresses()

            row = {
                'angle_deg': np.degrees(t),
                **forces,
                **efficiency,
                **stresses
            }
            data.append(row)

        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'swashplate_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
        )


if __name__ == "__main__":
    main()