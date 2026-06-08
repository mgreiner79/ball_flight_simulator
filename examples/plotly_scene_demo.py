import numpy as np

from ball_flight import BallPathSimulator, PitchParameters
from ball_flight_scene import CameraSimulator, SceneSimulator
from ball_flight_scene.adapters import trajectory_to_path3d
from ball_flight_scene.objects import Mound, Plate, StrikeZone


def main(renderer: str = "browser") -> None:
    scene = SceneSimulator()

    intrin_params = dict(
        f_mm=1.0,
        pixel_size_um=np.array([3, 3 * 480 / 400]),
        dimensions=np.array([640, 400]),
    )
    camera = CameraSimulator(
        **intrin_params,
        C=np.array([-2, 0.1, -2]),
        rotation=np.array([2, -3, 180]),
        length=15.5,
    )

    d_mound = 14.5
    simulator = BallPathSimulator()
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
    spin_trajectory = simulator.calculate(params, t_max=5, dt=0.005)
    no_spin_trajectory = BallPathSimulator().calculate(
        PitchParameters(**{**params.__dict__, "spin_rate": 0}),
        t_max=5,
        dt=0.005,
    )

    scene.add_camera(camera)
    scene.add_path(trajectory_to_path3d(spin_trajectory, "Ball Path with Spin"))
    scene.add_path(trajectory_to_path3d(no_spin_trajectory, "Ball Path without Spin"))
    scene.add_object(Mound(params.x0, 0))
    scene.add_object(Plate(d_mound, params.x0))
    scene.add_object(StrikeZone(d_mound, 0))

    scene.render_3d_scene(renderer=renderer)
    camera.render_projection(spin_trajectory.positions, renderer=renderer)


if __name__ == "__main__":
    main()
