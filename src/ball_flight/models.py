from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PitchParameters:
    """Initial conditions for a pitched ball trajectory.

    Coordinates are in meters, velocities are in meters per second, angles are
    in degrees, and spin rate is in revolutions per minute.
    """

    v0: float
    theta: float
    phi: float
    spin_rate: float
    spin_elevation: float
    spin_azimuth: float
    x0: float
    y0: float
    z0: float


@dataclass(frozen=True)
class Trajectory:
    """Time-aligned simulation result arrays."""

    time: FloatArray
    positions: FloatArray
    velocities: FloatArray
    acceleration: FloatArray
    drag_force: FloatArray
    spin_parameter: FloatArray
    lift_force: FloatArray
    lift_coefficient: FloatArray

    @property
    def x(self) -> FloatArray:
        return self.positions[:, 0]

    @property
    def y(self) -> FloatArray:
        return self.positions[:, 1]

    @property
    def z(self) -> FloatArray:
        return self.positions[:, 2]

    @property
    def vx(self) -> FloatArray:
        return self.velocities[:, 0]

    @property
    def vy(self) -> FloatArray:
        return self.velocities[:, 1]

    @property
    def vz(self) -> FloatArray:
        return self.velocities[:, 2]

    def to_array(self, include_time: bool = True) -> FloatArray:
        if include_time:
            return np.column_stack((self.positions, self.time))
        return self.positions.copy()
