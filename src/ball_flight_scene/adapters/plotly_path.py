from __future__ import annotations

import numpy as np

from ball_flight import Trajectory

from ..path import Path3d


def trajectory_to_path3d(trajectory: Trajectory, name: str = "Ball Path") -> Path3d:
    coords = np.array([trajectory.x, trajectory.y, trajectory.z])
    return Path3d(coords, name)
