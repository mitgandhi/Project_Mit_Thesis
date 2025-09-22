import unittest
import numpy as np
from calculate_dF import PistonGap, GeometryPistonGap, OperatingPistonGap
from mesh_calc_from_inp import generate_fields, setup_piston_gap


class TestPistonGap(unittest.TestCase):
    def setUp(self):
        self.N, self.M, self.Q = 30, 60, 20
        self.rK = 0.02  # 20mm radius
        self.L = 0.05  # 50mm length

        # Create test mesh
        r = np.linspace(0.98 * self.rK, 1.02 * self.rK, self.N)
        theta = np.linspace(0, 2 * np.pi, self.M)
        z = np.linspace(0, self.L, self.Q)
        theta_mg, r_mg, z_mg = np.meshgrid(theta, r, z)
        self.Theta = theta_mg

        # Setup piston gap
        self.piston = setup_piston_gap(self.N, self.M)

    def test_field_generation(self):
        """Test field generation and force calculations"""
        # Generate fields
        p, h1 = generate_fields(self.N, self.M, self.Q, self.Theta, self.piston.operating)

        # Test field shapes
        self.assertEqual(p.shape, (self.N, self.M, self.Q))
        self.assertEqual(h1.shape, (self.N, self.M, self.Q))

        # Set mid-plane values
        mid_Q = self.Q // 2
        self.piston.p = p[:, :, mid_Q]
        self.piston.h1 = h1[:, :, mid_Q]

        # Calculate forces
        self.piston.calc_fluid_forces()
        self.piston.calc_contact_forces()
        dF = self.piston.calc_force_difference()

        # Test force calculations
        self.assertNotEqual(self.piston.forces.FfK, 0)
        self.assertEqual(len(dF), 4)

        # Print results
        print("\nTest Results:")
        print(f"Max pressure: {np.max(p):.2e} Pa")
        print(f"Mean gap height: {np.mean(h1):.2e} m")
        print(f"Fluid force: {self.piston.forces.FfK:.2e} N")
        print(f"Contact force: {self.piston.forces.FcK:.2e} N")


if __name__ == '__main__':
    unittest.main()