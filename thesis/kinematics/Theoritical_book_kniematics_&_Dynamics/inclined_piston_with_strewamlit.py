import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec


class SwashplatePiston:
    def __init__(self, swash_angle, piston_angle, radius, pressure, piston_area, rpm, mk):
        """
        Initialize swashplate-piston calculator with SI units
        rb : It is the radis of cylinderblock pitch
        mk: mas of single piston-slipper assembly
        alpha: swashplate-angle
        phi: psiton angle with shaft axis
        """


        self.mk = mk
        self.alpha = np.radians(swash_angle)
        self.phi = np.radians(piston_angle)
        self.r = radius
        self.P = pressure
        self.A = piston_area
        self.omega = 2 * np.pi * rpm / 60

    def calculate_forces(self, theta):
        """Calculate forces at given rotation angle"""
        # Convert theta to degrees for checking range
        theta_deg = np.degrees(theta) % 360

        # If in 180-360 range, return zero forces
        if 180 <= theta_deg <= 360:
            return {
                "piston_force": 0,
                "Inertial_force": 0,
                "Reaction": 0,
                "displacement": self._calculate_displacement(
                    theta
                ),  # Keep displacement for visualization
                "velocity": self._calcualte_velocity(theta),
                "acceleration": self._calcualte_acc(theta),
            }


        # Pressure_force
        Fp = self.P * self.A * np.cos(self.phi)

        mu = np.sin(self.alpha)* np.sin(self.phi)/np.cos(self.alpha) *np.cos(self.phi)

        Fc1 = mu * Fp

        # Inertial forces
        Fi =  self.mk * self.omega **2 * self.r * np.tan(self.alpha)* np.tan(self.phi)  * np.cos(theta)

        # Reaction force
        Fr = (Fp+Fi-Fc1) / (np.cos(self.alpha)* np.cos(self.phi))

        # Kinematics calculations
        z = self._calculate_displacement(theta)
        v = self._calcualte_velocity(theta)
        a = self._calcualte_acc(theta)

        return {
            "piston_force": Fp,
            "Inertial_force": Fi,
            "Reaction": Fr,
            "displacement": z,
            "velocity": v,
            "acceleration": a,
        }

    def _calculate_displacement(self, theta):
        """Helper method to calculate displacement"""
        K1 = (self.r * (np.tan(self.alpha))) / (
            np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.alpha))
        )
        # K1= (self.r * (np.tan(self.alpha))) / (
        #     np.cos(self.phi))
        K2 = np.tan(self.phi) * np.tan(self.alpha)
        return (2 * K1) / (1 - K2)

    def _calcualte_velocity(self, theta):

        K1 = (self.r * (np.tan(self.alpha))) / (
            np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.alpha))
        )
        # K1 = (self.r * (np.tan(self.alpha))) / (
        #     np.cos(self.phi))
        K2 = np.tan(self.phi) * np.tan(self.alpha)
        return K1 * self.omega * (1 + K2) * np.sin(theta) / (1 + K2 * np.cos(theta))

    def _calcualte_acc(self, theta):

        K1 = (self.r * (np.tan(self.alpha))) / (
            np.cos(self.phi) * (1 + np.tan(self.phi) * np.tan(self.alpha))
        )
        # K1 = (self.r * (np.tan(self.alpha))) / (
        #     np.cos(self.phi))
        K2 = np.tan(self.phi) * np.tan(self.alpha)
        return (
            self.omega**2
            * K1
            * (1 + K2)
            * (K2 * (1 + np.sin(theta) * np.sin(theta)) + np.cos(theta))
            / ((1 + K2 * np.cos(theta)) ** 2)
        )

    def plot_force_analysis(self, revolutions=6):
        """Create plots with auto-scaling and selectable axes"""
        n_points = 360 * revolutions
        theta = np.linspace(0, 2 * np.pi * revolutions, n_points)
        results = [self.calculate_forces(t) for t in theta]

        # Available plotting options
        x_options = {
            "Rotation Angle [degrees]": np.degrees(theta),
            "Time [s]": np.linspace(0, revolutions * 60 / self.omega, n_points)
        }

        y_options = {
            "Piston Force [N]": ("piston_force", 'blue'),
            "Inertial Force [N]": ("Inertial_force", 'red'),
            "Reaction Force [N]": ("Reaction", 'green'),
            "Displacement [m]": ("displacement", 'purple'),
            "Velocity [m/s]": ("velocity", 'orange'),
            "Acceleration [m/s²]": ("acceleration", 'brown')
        }

        # Create Streamlit controls
        st.sidebar.markdown("### Plot Controls")
        x_selection = st.sidebar.selectbox("X-Axis", list(x_options.keys()))
        y_selections = st.sidebar.multiselect(
            "Y-Axes",
            list(y_options.keys()),
            default=list(y_options.keys())[0]
        )

        # Create plot
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)

        # Plot selected data and track max values
        max_val = float('-inf')
        min_val = float('inf')

        for y_sel in y_selections:
            y_key, color = y_options[y_sel]
            y_values = [r[y_key] for r in results]
            ax.plot(x_options[x_selection], y_values, color=color, label=y_sel, linewidth=2.5)

            # Update min/max values
            max_val = max(max_val, max(y_values))
            min_val = min(min_val, min(y_values))

        # Set y-axis limits with 10% padding
        padding = (max_val - min_val) * 0.1
        ax.set_ylim(min_val - padding, max_val + padding)

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel(x_selection, fontsize=12, labelpad=10)
        ax.set_ylabel("Value", fontsize=12, labelpad=10)
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Format large numbers
        if abs(max_val) > 1000 or abs(min_val) > 1000:
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
            ax.yaxis.get_major_formatter().set_scientific(True)
            ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

        plt.tight_layout()
        return fig

    def create_3d_visualization(self, revolutions=2):
        """Create 3D visualization of piston motion"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Set up trajectory
        n_points = 360 * revolutions
        theta = np.linspace(0, 2 * np.pi * revolutions, n_points)
        z = np.array([self.calculate_forces(t)["displacement"] for t in theta])
        x = self.r * np.cos(theta)
        y = self.r * np.sin(theta)

        # Plot trajectory
        ax.plot(x, y, z, "b-", alpha=0.3, label="Piston Path")

        # Create swashplate surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, self.r, 20)
        U, V = np.meshgrid(u, v)
        X = V * np.cos(U)
        Y = V * np.sin(U)
        Z = V * np.tan(self.alpha) * np.cos(U)

        # Plot swashplate
        ax.plot_surface(X, Y, Z, alpha=0.4, color="gray")

        # Set axis labels and title
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Inclined Piston Motion on Swashplate")
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        return fig


class SwashplateAnalyzer(SwashplatePiston):
    def calculate_efficiency(self):
        """Calculate system efficiencies"""
        friction_coefficient = 0.1
        mechanical_loss = friction_coefficient * (np.tan(self.alpha + self.phi))
        mechanical_efficiency = 1 - mechanical_loss

        leakage_factor = 0.05
        volumetric_efficiency = 1 - leakage_factor * (self.P / 1e6) / 100

        total_efficiency = mechanical_efficiency * volumetric_efficiency

        theoretical_power = (
            self.P * self.A * self.r * np.tan(self.alpha + self.phi) * self.omega
        )
        actual_power = theoretical_power * total_efficiency

        return {
            "mechanical_efficiency": mechanical_efficiency,
            "volumetric_efficiency": volumetric_efficiency,
            "total_efficiency": total_efficiency,
            "theoretical_power": theoretical_power,
            "actual_power": actual_power,
        }

    def calculate_stresses(self):
        """Calculate critical component stresses"""
        bearing_load = self.P * self.A / np.cos(self.alpha)*np.tan(self.phi)
        bearing_area = np.pi * self.r * 0.1
        bearing_stress = bearing_load / bearing_area

        piston_stress = self.P

        surface_stress = self.P * self.A * np.tan(self.alpha) / (np.pi * self.r**2)

        return {
            "bearing_stress": bearing_stress,
            "piston_stress": piston_stress,
            "surface_stress": surface_stress,
        }


# Replace the params section with this:
def validate_float_input(value, min_val, max_val, default):
    """Validate float input and ensure it's within bounds"""
    try:
        val = float(value)
        if min_val <= val <= max_val:
            return val
        return default
    except:
        return default


def main():
    st.set_page_config(page_title="Swashplate Analysis System", layout="wide")
    st.title("Swashplate Analysis System")

    # Sidebar for parameters
    st.sidebar.title("Parameters")

    # Presets
    presets = {
        "High Speed": {
            "swash_angle": 15,
            "piston_angle": 5,
            "radius": 0.07837,
            "pressure": 250e6,
            "piston_area": 0.0003529,
            "rpm": 6000,
            "mk": 164,
        },
        "High Pressure": {
            "swash_angle": 15,
            "piston_angle": 8,
            "radius": 0.06,
            "pressure": 35e6,
            "piston_area": 0.0012,
            "rpm": 1200,
            "mk": 164,
        },
        "Balanced": {
            "swash_angle": 12,
            "piston_angle": 7,
            "radius": 0.05,
            "pressure": 20e6,
            "piston_area": 0.001,
            "rpm": 3000,
            "mk":164,
        },
    }

    preset = st.sidebar.selectbox("Select Preset", ["Custom"] + list(presets.keys()))

    if preset == "Custom":
        st.sidebar.markdown("### Parameter Controls")

        # Swashplate Angle
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            swash_angle_slider = st.slider("Swashplate Angle [°]", 0, 30, 15)
        with col2:
            swash_angle_text = st.text_input(
                "", value=str(swash_angle_slider), key="swash_text"
            )
            swash_angle = validate_float_input(
                swash_angle_text, 0, 30, swash_angle_slider
            )

        # Piston Angle
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            piston_angle_slider = st.slider("Piston Angle [°]", 0, 30, 10)
        with col2:
            piston_angle_text = st.text_input(
                "", value=str(piston_angle_slider), key="piston_text"
            )
            piston_angle = validate_float_input(
                piston_angle_text, 0, 30, piston_angle_slider
            )

        # Radius
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            radius_slider = st.slider("Radius [mm]", 15, 100, 50)
        with col2:
            radius_text = st.text_input("", value=str(radius_slider), key="radius_text")
            radius = (
                validate_float_input(radius_text, 15, 100, radius_slider) / 1000
            )  # Convert to meters

        # Pressure
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            pressure_slider = st.slider("Pressure [MPa]", 1, 40, 20)
        with col2:
            pressure_text = st.text_input(
                "", value=str(pressure_slider), key="pressure_text"
            )
            pressure = (
                validate_float_input(pressure_text, 1, 40, pressure_slider) * 1e6
            )  # Convert to Pa

        # Piston Area
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            area_slider = st.slider("Piston Area [mm²]", 100, 2000, 1000)
        with col2:
            area_text = st.text_input("", value=str(area_slider), key="area_text")
            piston_area = (
                validate_float_input(area_text, 100, 2000, area_slider) / 1e6
            )  # Convert to m²

        # RPM
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            rpm_slider = st.slider("Speed [RPM]", 0, 6000, 1500)
        with col2:
            rpm_text = st.text_input("", value=str(rpm_slider), key="rpm_text")
            rpm = validate_float_input(rpm_text, 0, 6000, rpm_slider)


        # MASS_SLIPPER-PISTON ASSEMBLY
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            mass = st.slider("Piston Area [mm²]", 100, 2000, 164)
        with col2:
            mass_text = st.text_input("", value=str(mass), key="mass_text")
            p_s_mass = (
                    validate_float_input(mass_text, 100, 2000, area_slider) / 1000
            )  # Convert to m²

        params = {
            "swash_angle": swash_angle,
            "piston_angle": piston_angle,
            "radius": radius,
            "pressure": pressure,
            "piston_area": piston_area,
            "rpm": rpm,
            "mk": p_s_mass,
        }

        # Optional: Display current values in a more readable format
        st.sidebar.markdown("### Current Values")
        st.sidebar.write(f"Swashplate Angle: {params['swash_angle']:.2f}°")
        st.sidebar.write(f"Piston Angle: {params['piston_angle']:.2f}°")
        st.sidebar.write(f"Radius: {params['radius'] * 1000:.2f} mm")
        st.sidebar.write(f"Pressure: {params['pressure'] / 1e6:.2f} MPa")
        st.sidebar.write(f"Piston Area: {params['piston_area'] * 1e6:.2f} mm²")
        st.sidebar.write(f"Speed: {params['rpm']:.0f} RPM")
        st.sidebar.write(f"P_S_mass: {params['mk']:.0f} Kg")

    # Create analyzer instance
    analyzer = SwashplateAnalyzer(**params)

    # Calculate results
    efficiency_results = analyzer.calculate_efficiency()
    # stress_results = analyzer.calculate_stresses()

    # Display results in columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Efficiency Analysis")
        st.write(
            f"Mechanical Efficiency: {efficiency_results['mechanical_efficiency']:.2%}"
        )
        st.write(
            f"Volumetric Efficiency: {efficiency_results['volumetric_efficiency']:.2%}"
        )
        st.write(f"Total Efficiency: {efficiency_results['total_efficiency']:.2%}")
        st.write(f"Actual Power: {efficiency_results['actual_power'] / 1000:.1f} kW")

    # with col2:
    #     st.subheader("Stress Analysis")
    #     st.write(f"Bearing Stress: {stress_results['bearing_stress'] / 1e6:.1f} MPa")
    #     st.write(f"Piston Stress: {stress_results['piston_stress'] / 1e6:.1f} MPa")
    #     st.write(f"Surface Stress: {stress_results['surface_stress'] / 1e6:.1f} MPa")

    # Display plots
    st.subheader("Force Analysis")
    force_fig = analyzer.plot_force_analysis()
    st.pyplot(force_fig)

    st.subheader("3D Visualization")
    visual_fig = analyzer.create_3d_visualization()
    st.pyplot(visual_fig)


if __name__ == "__main__":
    main()
