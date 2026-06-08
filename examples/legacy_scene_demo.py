"""Compatibility demo entry point.

The maintained version of this demo lives in ``examples/plotly_scene_demo.py``.
"""

import numpy as np

from ball_flight import BallPathSimulator, PitchParameters
from ball_flight_scene import CameraSimulator, SceneSimulator
from ball_flight_scene.adapters import trajectory_to_path3d
from ball_flight_scene.objects import Mound, Plate, StrikeZone


scene = SceneSimulator()

intrin_params = dict(
    f_mm=1.0,
    pixel_size_um=np.array([3, 3 * 480 / 400]),
    dimensions=np.array([640, 400]),
)
pos = dict(cx=-2, cy=0.1, cz=-2)
rot = dict(yaw=2, pitch=-3, roll=180)

cam = CameraSimulator(
    **intrin_params,
    C=np.array([v for v in pos.values()]),
    rotation=np.array([v for v in rot.values()]),
    length=15.5,
)

baseline = 0.076
cam1 = CameraSimulator(
    **intrin_params,
    C=np.array([pos["cx"], pos["cy"], pos["cz"] + baseline]),
    rotation=np.array([v for v in rot.values()]),
)

d_mound = 14.5
params = PitchParameters(
    v0=42,
    theta=0,
    phi=2,
    spin_rate=1200,
    spin_elevation=180,
    spin_azimuth=0,
    x0=-0.5,
    y0=1.6,
    z0=0,
)

sim = BallPathSimulator("Ball Path with Spin")
trajectory = sim.calculate(params, t_max=5, dt=0.005)
trajectory0 = BallPathSimulator("Ball Path without Spin").calculate(
    PitchParameters(**{**params.__dict__, "spin_rate": 0}),
    t_max=5,
    dt=0.005,
)

scene.add_camera(cam)
scene.add_camera(cam1)
scene.add_path(trajectory_to_path3d(trajectory, "Ball Path with Spin"))
scene.add_path(trajectory_to_path3d(trajectory0, "Ball Path without Spin"))
scene.add_object(Mound(params.x0, 0))
scene.add_object(Plate(d_mound, params.x0))
scene.add_object(StrikeZone(d_mound, 0))

points = trajectory.positions
plate_idx = min(range(len(trajectory.x)), key=lambda i: abs(trajectory.x[i] - (d_mound - params.x0)))
break_h = trajectory.z[plate_idx] - trajectory0.z[plate_idx]
break_v = trajectory.y[plate_idx] - trajectory0.y[plate_idx]
final_speed = np.linalg.norm(trajectory.velocities[-1])
final_speed0 = np.linalg.norm(trajectory0.velocities[-1])


if __name__ == "__main__":
    scene.render_3d_scene()
    cam.render_projection(points)
    scene.render_projected_scene()
