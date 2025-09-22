import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GeometryPistonGap:
    mK: float  # Mass piston/slipper assembly
    lSK: float  # Distance piston center of mass/slipper assembly from piston head
    zRK: float  # Distance between piston head and beginning of the gap
    lvar: float  # Variable length
    hmin: float  # Minimum gap height


@dataclass
class OperatingPistonGap:
    beta_rad: float  # Swashplate angle
    betamax_rad: float  # Max swashplate angle
    gamma_rad: float  # Swashplate cross angle
    phi_rad: float  # Angular position
    omega: float  # Angular speed
    pDC: float  # Pressure displacement chamber
    pCase: float  # Case pressure
    vK: float  # Piston velocity


@dataclass
class ForcesPistonGap:
    # Initialize all forces with default values
    FfK: float = 0.0
    FfKx: float = 0.0
    FfKy: float = 0.0
    MfKx: float = 0.0
    MfKy: float = 0.0
    FcK: float = 0.0
    FcKx: float = 0.0
    FcKy: float = 0.0
    McKx: float = 0.0
    McKy: float = 0.0
    FK: float = 0.0
    FKx: float = 0.0
    FKy: float = 0.0
    MK: float = 0.0
    MKx: float = 0.0
    MKy: float = 0.0
    FTG: float = 0.0
    Fsk: float = 0.0
    MTK: float = 0.0
    McK: float = 0.0
    MfK: float = 0.0

    # Initialize lists for force components
    F_fluid: List[float] = None
    F_contact: List[float] = None
    F_external: List[float] = None
    dF: List[float] = None

    def __post_init__(self):
        # Initialize lists with zeros
        self.F_fluid = [0.0] * 4
        self.F_contact = [0.0] * 4
        self.F_external = [0.0] * 4
        self.dF = [0.0] * 4


class PistonGap:
    def __init__(self, geometry: GeometryPistonGap, operating: OperatingPistonGap):
        self.geometry = geometry
        self.operating = operating
        self.forces = ForcesPistonGap()

        # Initialize properties with default values
        self.force_balance_iterative = False
        self.N = 0
        self.M = 0
        self.Q = 0
        self.AreaK = 0.0
        self.AreaG = 0.0
        self.rB = 0.0
        self.rK = 0.0
        self.rZ = 0.0
        self.Eprime = 2.1e11  # Elastic modulus
        self.muT = 0.01  # Viscosity

    def initialize_mesh(self, N: int, M: int, Q: int):
        """Initialize computational mesh and arrays"""
        self.N = N
        self.M = M
        self.Q = Q

        # Initialize mesh arrays with proper shapes
        self.p = np.zeros((N, M))
        self.h1 = np.ones((N, M)) * 10e-6  # Initial gap height
        self.sigma = np.zeros((N, M))
        self.dAz = np.ones((N, M)) * 1e-6  # Area elements
        self.zKj = np.ones((N, M)) * 0.01  # Z-coordinate
        self.phi = np.ones((N, M)) * np.pi / 4  # Angular position

        # Initialize velocity fields with proper shapes
        self.vx_p = np.zeros((Q, N, M))
        self.vy_p = np.zeros((Q, N, M))
        self.vx_c = np.zeros((Q, N, M))
        self.vy_c = np.zeros((Q, N, M))
        self.dvxz = np.zeros((Q, N, M))
        self.dvyz = np.zeros((Q, N, M))

        # Initialize grid spacing
        self.dx = 1.0 / (N - 1)
        self.dy = 1.0 / (M - 1)
        self.dz2 = np.ones((N, M)) * (1.0 / (Q - 1))

    def calc_fluid_forces(self):
        """Calculate fluid forces from fluid pressure"""
        # Calculate total fluid force
        self.forces.FfK = float(np.sum(self.p * self.dAz))

        # Calculate mean values for calculations
        mean_phi = np.mean(self.phi)
        mean_zKj = np.mean(self.zKj)

        # Calculate forces and moments
        self.forces.FfKx = float(-1.0 * self.forces.FfK * np.cos(mean_phi))
        self.forces.FfKy = float(-1.0 * self.forces.FfK * np.sin(mean_phi))
        self.forces.MfKx = float(-1.0 * self.forces.FfKy * mean_zKj)
        self.forces.MfKy = float(self.forces.FfKx * mean_zKj)

        # Calculate magnitudes
        self.forces.FfK = float(np.sqrt(self.forces.FfKx ** 2 + self.forces.FfKy ** 2))
        self.forces.MfK = float(np.sqrt(self.forces.MfKx ** 2 + self.forces.MfKy ** 2))

    def calc_contact_forces(self):
        """Calculate contact forces from elastic compenetration"""
        # Calculate contact stress
        if self.force_balance_iterative:
            self.sigma += 0.1 * self.Eprime * (self.geometry.hmin - self.h1) / self.rK
            self.sigma = np.where(self.sigma < 0.0, 0.0, self.sigma)
            self.sigma = np.where(self.sigma > 288e6, 288e6, self.sigma)
        else:
            # Add small tolerance to prevent division by zero
            small_tol = 1e-10
            self.sigma = np.where(self.h1 < self.geometry.hmin,
                                  self.Eprime * (self.geometry.hmin - self.h1) / (self.rK + small_tol),
                                  0.0)

        # Calculate total contact force
        self.forces.FcK = float(np.sum(self.sigma * self.dAz))

        # Calculate mean values for calculations
        mean_phi = np.mean(self.phi)
        mean_zKj = np.mean(self.zKj)

        # Calculate forces and moments
        self.forces.FcKx = float(-1.0 * self.forces.FcK * np.cos(mean_phi))
        self.forces.FcKy = float(-1.0 * self.forces.FcK * np.sin(mean_phi))
        self.forces.McKx = float(-1.0 * self.forces.FcKy * mean_zKj)
        self.forces.McKy = float(self.forces.FcKx * mean_zKj)

        # Calculate magnitudes
        self.forces.FcK = float(np.sqrt(self.forces.FcKx ** 2 + self.forces.FcKy ** 2))
        self.forces.McK = float(np.sqrt(self.forces.McKx ** 2 + self.forces.McKy ** 2))

        # Calculate control point forces
        self.forces.F_contact[2] = float(self.forces.McKy / self.geometry.lvar)
        self.forces.F_contact[3] = float(-1.0 * self.forces.McKx / self.geometry.lvar)
        self.forces.F_contact[0] = float(-1.0 * self.forces.F_contact[2] + self.forces.FcKx)
        self.forces.F_contact[1] = float(-1.0 * self.forces.F_contact[3] + self.forces.FcKy)

    def calc_force_difference(self) -> List[float]:
        """Calculate force difference on control points"""
        if not self.force_balance_iterative:
            self.calc_contact_forces()

        # Calculate fluid forces in control points
        self.forces.F_fluid[2] = float(self.forces.MfKy / self.geometry.lvar)
        self.forces.F_fluid[3] = float(-1.0 * self.forces.MfKx / self.geometry.lvar)
        self.forces.F_fluid[0] = float(-1.0 * self.forces.F_fluid[2] + self.forces.FfKx)
        self.forces.F_fluid[1] = float(-1.0 * self.forces.F_fluid[3] + self.forces.FfKy)

        # Calculate external forces in control points
        self.forces.F_external[2] = float(self.forces.MKy / self.geometry.lvar)
        self.forces.F_external[3] = float(-1.0 * self.forces.MKx / self.geometry.lvar)
        self.forces.F_external[0] = float(-1.0 * self.forces.F_external[2] + self.forces.FKx)
        self.forces.F_external[1] = float(-1.0 * self.forces.F_external[3] + self.forces.FKy)

        # Calculate force differences
        for j in range(4):
            self.forces.dF[j] = float(-self.forces.F_external[j] -
                                      self.forces.F_fluid[j] -
                                      self.forces.F_contact[j])

        return self.forces.dF