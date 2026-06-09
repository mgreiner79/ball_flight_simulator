"""Render tutorial diagrams as PyVista scenes.

Run from the repository root:

    .\.venv\Scripts\python.exe docs\render_tutorial_images.py

The generated PNGs are committed documentation assets. The script is kept so
the figures can be adjusted or regenerated as the simulator evolves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


ASSET_DIR = Path(__file__).resolve().parent / "assets"
WINDOW_SIZE = (1500, 950)

WHITE = "#ffffff"
BLACK = "#111827"
GRAY = "#6b7280"
LIGHT_GRAY = "#e5e7eb"
BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
PURPLE = "#7c3aed"
ORANGE = "#f97316"
YELLOW = "#facc15"


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def make_plotter() -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background(WHITE)
    plotter.enable_anti_aliasing("ssaa")
    return plotter


def add_arrow(
    plotter: pv.Plotter,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    color: str,
    shaft_radius: float = 0.035,
    tip_radius: float = 0.055,
    tip_length: float = 0.125,
) -> None:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    direction = end_array - start_array
    length = np.linalg.norm(direction)
    arrow = pv.Arrow(
        start=start_array,
        direction=unit(direction),
        scale=length,
        shaft_radius=shaft_radius,
        tip_radius=tip_radius,
        tip_length=tip_length,
    )
    plotter.add_mesh(arrow, color=color, smooth_shading=True)


def add_tube(
    plotter: pv.Plotter,
    points: list[tuple[float, float, float]],
    color: str,
    radius: float = 0.025,
) -> None:
    line = pv.Spline(np.asarray(points, dtype=float), len(points) * 20)
    plotter.add_mesh(line.tube(radius=radius), color=color, smooth_shading=True)


def add_polyline_tube(
    plotter: pv.Plotter,
    points: np.ndarray,
    color: str,
    radius: float = 0.018,
) -> None:
    polyline = pv.PolyData(points)
    polyline.lines = np.hstack(([len(points)], np.arange(len(points))))
    plotter.add_mesh(polyline.tube(radius=radius), color=color, smooth_shading=True)


def add_angle_arc(
    plotter: pv.Plotter,
    center: tuple[float, float, float],
    axis_1: tuple[float, float, float],
    axis_2: tuple[float, float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
    color: str,
    tube_radius: float = 0.018,
) -> np.ndarray:
    angles = np.radians(np.linspace(start_deg, end_deg, 80))
    center_array = np.asarray(center, dtype=float)
    u1 = unit(np.asarray(axis_1, dtype=float))
    u2 = unit(np.asarray(axis_2, dtype=float))
    points = center_array + radius * (
        np.cos(angles)[:, None] * u1 + np.sin(angles)[:, None] * u2
    )
    add_polyline_tube(plotter, points, color, radius=tube_radius)
    return points


def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis_unit = unit(axis)
    reference = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(axis_unit, reference)) > 0.92:
        reference = np.array([1.0, 0.0, 0.0])
    b1 = unit(np.cross(axis_unit, reference))
    b2 = unit(np.cross(axis_unit, b1))
    return b1, b2


def add_curved_spin_arrow(
    plotter: pv.Plotter,
    axis_start: tuple[float, float, float],
    axis_end: tuple[float, float, float],
    color: str,
    radius: float = 0.34,
    start_deg: float = 25,
    end_deg: float = 285,
) -> None:
    start = np.asarray(axis_start, dtype=float)
    end = np.asarray(axis_end, dtype=float)
    axis = end - start
    axis_unit = unit(axis)
    b1, b2 = perpendicular_basis(axis_unit)

    center = start + axis_unit * (np.linalg.norm(axis) * 0.62)
    arc_points = add_angle_arc(
        plotter,
        tuple(center),
        tuple(b1),
        tuple(b2),
        radius,
        start_deg,
        end_deg,
        color,
        tube_radius=0.025,
    )

    tangent = unit(arc_points[-1] - arc_points[-2])
    cone_height = 0.18
    cone = pv.Cone(
        center=arc_points[-1] + tangent * (cone_height / 2),
        direction=tangent,
        height=cone_height,
        radius=0.07,
        resolution=48,
    )
    plotter.add_mesh(cone, color=color, smooth_shading=True)


def add_text_label(
    plotter: pv.Plotter,
    text: str,
    position: tuple[float, float, float],
    font_size: int = 30,
    color: str = BLACK,
) -> None:
    plotter.add_point_labels(
        [position],
        [text],
        font_size=font_size,
        text_color=color,
        show_points=False,
        point_color=color,
        point_size=0,
        shape=None,
        always_visible=True,
        render_points_as_spheres=False,
    )


def add_overlay_text(
    plotter: pv.Plotter,
    text: str,
    position: str | tuple[int, int] = "upper_left",
    font_size: int = 22,
    color: str = BLACK,
) -> None:
    plotter.add_text(text, position=position, font_size=font_size, color=color, font="arial")


def add_baseball(plotter: pv.Plotter, center: tuple[float, float, float], radius: float = 0.26) -> None:
    sphere = pv.Sphere(radius=radius, center=center, theta_resolution=80, phi_resolution=40)
    plotter.add_mesh(sphere, color="#f8fafc", smooth_shading=True, specular=0.25)

    cx, cy, cz = center
    seam_1 = [(cx + 0.02 * np.sin(t), cy + radius * 0.70 * np.sin(t), cz + radius * 0.82 * np.cos(t)) for t in np.linspace(0, 2 * np.pi, 140)]
    seam_2 = [(cx + radius * 0.72 * np.sin(t), cy + 0.02 * np.cos(t), cz + radius * 0.82 * np.cos(t)) for t in np.linspace(0, 2 * np.pi, 140)]
    add_tube(plotter, seam_1, RED, radius=0.008)
    add_tube(plotter, seam_2, RED, radius=0.008)


def add_ground(plotter: pv.Plotter, center=(0, -0.02, 0), size=8.0) -> None:
    ground = pv.Plane(center=center, direction=(0, 1, 0), i_size=size, j_size=size)
    plotter.add_mesh(ground, color="#f3f4f6", opacity=1.0)


def set_camera(plotter: pv.Plotter, position, focal_point, view_up=(0, 1, 0), parallel_scale: float | None = None) -> None:
    plotter.camera_position = (position, focal_point, view_up)
    if parallel_scale is not None:
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = parallel_scale
    else:
        plotter.camera.zoom(1.05)


def render_coordinate_system() -> None:
    plotter = make_plotter()
    add_ground(plotter, center=(2.1, -0.02, 1.7), size=7.5)

    release = np.array((0.0, 0.55, 0.0))
    plate_center = np.array((0.0, 0.02, 4.2))
    add_baseball(plotter, tuple(release), radius=0.16)

    plate = pv.Box(bounds=(-0.32, 0.32, 0.0, 0.03, 4.0, 4.45))
    plotter.add_mesh(plate, color="#f9fafb", smooth_shading=True)

    add_arrow(plotter, tuple(release), (1.45, 0.55, 0.0), RED)
    add_arrow(plotter, tuple(release), (0.0, 2.0, 0.0), GREEN)
    add_arrow(plotter, tuple(release), (0.0, 0.55, 2.2), BLUE)
    add_tube(plotter, [(0.0, 0.55, 0.0), (0.15, 0.82, 1.4), (0.05, 0.72, 3.25), tuple(plate_center + (0, 0.5, 0))], GRAY, radius=0.018)

    add_text_label(plotter, "+X", (1.62, 0.72, -0.20), color=RED, font_size=24)
    add_text_label(plotter, "+Y", (0.10, 2.05, 0.00), color=GREEN, font_size=24)
    add_text_label(plotter, "+Z", (0.05, 1.06, 2.38), color=BLUE, font_size=24)
    add_text_label(plotter, "r0", (-0.25, 0.88, -0.20), color=BLACK, font_size=22)
    add_text_label(plotter, "P", (-0.50, 0.20, 4.58), color=BLACK, font_size=22)
    add_text_label(plotter, "y=0", (1.55, 0.12, -0.55), color=GRAY, font_size=21)
    add_text_label(plotter, "r(t)", (0.45, 0.98, 1.72), color=GRAY, font_size=21)

    plotter.add_light(pv.Light(position=(2, 4, -3), focal_point=(0, 0.5, 2), intensity=0.8))
    set_camera(plotter, (4.2, 3.0, -4.4), (0.25, 0.75, 2.0), parallel_scale=2.7)
    plotter.screenshot(ASSET_DIR / "coordinate_system_pyvista.png")
    plotter.close()


def render_polar_coordinates() -> None:
    plotter = make_plotter()
    origin = np.array((0.0, 0.0, 0.0))
    velocity = np.array((1.15, 0.9, 2.05))
    projection = np.array((velocity[0], 0.0, velocity[2]))
    theta_deg = np.degrees(np.arctan2(velocity[1], np.linalg.norm(projection)))
    phi_deg = np.degrees(np.arctan2(velocity[0], velocity[2]))

    add_baseball(plotter, tuple(origin), radius=0.13)
    add_arrow(plotter, tuple(origin), (1.35, 0.0, 0.0), RED, shaft_radius=0.022)
    add_arrow(plotter, tuple(origin), (0.0, 1.35, 0.0), GREEN, shaft_radius=0.022)
    add_arrow(plotter, tuple(origin), (0.0, 0.0, 1.55), BLUE, shaft_radius=0.022)
    add_arrow(plotter, tuple(origin), tuple(velocity), PURPLE, shaft_radius=0.028)
    add_tube(plotter, [tuple(origin), tuple(projection)], GRAY, radius=0.014)
    add_tube(plotter, [tuple(projection), tuple(velocity)], LIGHT_GRAY, radius=0.012)

    projection_unit = unit(projection)
    add_angle_arc(plotter, tuple(origin), (0, 0, 1), (1, 0, 0), 0.55, 0, phi_deg, ORANGE)
    add_angle_arc(plotter, tuple(origin), tuple(projection_unit), (0, 1, 0), 0.82, 0, theta_deg, GREEN)

    add_text_label(plotter, "v0", tuple(velocity + np.array((0.06, 0.16, 0.04))), color=PURPLE, font_size=25)
    add_text_label(plotter, "vh", tuple(projection * 0.58 + np.array((-0.10, -0.18, 0.00))), color=GRAY, font_size=22)
    add_text_label(plotter, "ph", (0.18, 0.10, 0.62), color=ORANGE, font_size=25)
    add_text_label(plotter, "th", tuple(projection_unit * 0.52 + np.array((-0.16, 0.40, 0.06))), color=GREEN, font_size=25)
    add_text_label(plotter, "+X", (1.48, 0.08, -0.02), color=RED, font_size=22)
    add_text_label(plotter, "+Y", (0.06, 1.50, 0.02), color=GREEN, font_size=22)
    add_text_label(plotter, "+Z", (0.04, 0.08, 1.72), color=BLUE, font_size=22)
    add_overlay_text(plotter, "Launch polar angles", position=(60, 840), font_size=22)

    plotter.add_light(pv.Light(position=(2, 3, -4), focal_point=(0.5, 0.4, 0.8), intensity=0.85))
    set_camera(plotter, (3.5, 2.4, -4.2), (0.55, 0.45, 0.8), parallel_scale=2.1)
    plotter.screenshot(ASSET_DIR / "polar_coordinates_pyvista.png")
    plotter.close()


def render_forces_on_ball() -> None:
    plotter = make_plotter()
    add_baseball(plotter, (0, 0, 0), radius=0.38)

    add_arrow(plotter, (0, 0, 0), (1.85, 0.20, 0.0), BLUE, shaft_radius=0.03)
    add_arrow(plotter, (0, 0, 0), (-1.45, -0.16, 0.0), RED, shaft_radius=0.03)
    add_arrow(plotter, (0, 0, 0), (0, -1.75, 0), GREEN, shaft_radius=0.03)
    add_arrow(plotter, (0, 0, 0), (0.55, 1.55, 0.0), PURPLE, shaft_radius=0.03)

    add_text_label(plotter, "v", (2.02, 0.34, 0.0), color=BLUE, font_size=27)
    add_text_label(plotter, "F_d", (-1.82, -0.30, 0.0), color=RED, font_size=27)
    add_text_label(plotter, "mg", (0.12, -1.93, 0.0), color=GREEN, font_size=27)
    add_text_label(plotter, "F_l", (0.74, 1.72, 0.0), color=PURPLE, font_size=27)
    add_overlay_text(plotter, "F_net = F_drag + F_lift + F_gravity\n a = F_net / m", position=(70, 790), font_size=20)

    plotter.add_light(pv.Light(position=(2, 3, -4), focal_point=(0, 0, 0), intensity=0.9))
    set_camera(plotter, (0, 0, 6), (0.0, 0.0, 0.0), parallel_scale=2.55)
    plotter.screenshot(ASSET_DIR / "forces_on_ball_pyvista.png")
    plotter.close()


def render_spin_decomposition() -> None:
    plotter = make_plotter()
    add_baseball(plotter, (0, 0, 0), radius=0.22)

    origin = (0, 0, 0)
    velocity_end = (2.8, 0, 0)
    omega_end = (2.0, 1.25, 0.85)
    parallel_end = (2.0, 0, 0)

    add_arrow(plotter, origin, velocity_end, BLUE, shaft_radius=0.025)
    add_arrow(plotter, origin, omega_end, PURPLE, shaft_radius=0.028)
    add_arrow(plotter, origin, parallel_end, RED, shaft_radius=0.023)
    add_arrow(plotter, parallel_end, omega_end, GREEN, shaft_radius=0.023)
    add_tube(plotter, [omega_end, parallel_end], LIGHT_GRAY, radius=0.01)

    add_text_label(plotter, "u_v", (2.86, 0.14, 0.05), color=BLUE, font_size=24)
    add_text_label(plotter, "w", (1.78, 1.42, 0.95), color=PURPLE, font_size=28)
    add_text_label(plotter, "wP", (1.02, -0.30, 0.0), color=RED, font_size=24)
    add_text_label(plotter, "wT", (1.40, 1.02, 0.38), color=GREEN, font_size=24)
    add_overlay_text(plotter, "omega_perp = omega - (omega dot u_v) u_v", position=(60, 845), font_size=20)

    plotter.add_light(pv.Light(position=(2, 4, -3), focal_point=(0.7, 0.4, 0.2), intensity=0.9))
    set_camera(plotter, (4.2, 2.8, -4.0), (0.9, 0.45, 0.25), parallel_scale=2.55)
    plotter.screenshot(ASSET_DIR / "spin_decomposition_pyvista.png")
    plotter.close()


def render_spin_axis_polar_coordinates() -> None:
    plotter = make_plotter()
    origin = np.array((0.0, 0.0, 0.0))
    u_v_end = np.array((0.0, 0.0, 1.75))
    u_perp1_end = np.array((1.25, 0.0, 0.0))
    u_perp2_end = np.array((0.0, 1.25, 0.0))
    omega_end = np.array((0.85, 1.0, 1.25))
    horizontal_projection = np.array((omega_end[0], 0.0, omega_end[2]))
    spin_elevation_deg = np.degrees(
        np.arctan2(omega_end[1], np.linalg.norm(horizontal_projection))
    )
    spin_azimuth_deg = np.degrees(np.arctan2(omega_end[2], omega_end[0]))

    add_baseball(plotter, tuple(origin), radius=0.13)
    add_arrow(plotter, tuple(origin), tuple(u_v_end), BLUE, shaft_radius=0.022)
    add_arrow(plotter, tuple(origin), tuple(u_perp1_end), RED, shaft_radius=0.022)
    add_arrow(plotter, tuple(origin), tuple(u_perp2_end), GREEN, shaft_radius=0.022)
    add_arrow(plotter, tuple(origin), tuple(omega_end), PURPLE, shaft_radius=0.03)
    add_tube(plotter, [tuple(origin), tuple(horizontal_projection)], GRAY, radius=0.014)
    add_tube(plotter, [tuple(horizontal_projection), tuple(omega_end)], LIGHT_GRAY, radius=0.012)
    add_curved_spin_arrow(plotter, tuple(origin), tuple(omega_end), YELLOW)

    horizontal_unit = unit(horizontal_projection)
    add_angle_arc(plotter, tuple(origin), (1, 0, 0), (0, 0, 1), 0.55, 0, spin_azimuth_deg, ORANGE)
    add_angle_arc(plotter, tuple(origin), tuple(horizontal_unit), (0, 1, 0), 0.82, 0, spin_elevation_deg, GREEN)

    add_text_label(plotter, "w", tuple(omega_end + np.array((0.08, 0.12, 0.08))), color=PURPLE, font_size=28)
    add_text_label(plotter, "spin", tuple(omega_end * 0.62 + np.array((-0.30, 0.46, 0.16))), color=YELLOW, font_size=22)
    add_text_label(plotter, "az", (0.46, 0.05, 0.42), color=ORANGE, font_size=25)
    add_text_label(plotter, "el", tuple(horizontal_unit * 0.62 + np.array((-0.10, 0.34, 0.02))), color=GREEN, font_size=25)
    add_text_label(plotter, "u1", tuple(u_perp1_end + np.array((0.12, 0.04, 0.0))), color=RED, font_size=22)
    add_text_label(plotter, "u2", tuple(u_perp2_end + np.array((0.04, 0.12, 0.0))), color=GREEN, font_size=22)
    add_text_label(plotter, "u_v", (0.35, -0.24, 1.45), color=BLUE, font_size=20)
    add_overlay_text(plotter, "Spin-axis polar coordinates", position=(60, 845), font_size=22)

    plotter.add_light(pv.Light(position=(2, 4, -3), focal_point=(0.4, 0.5, 0.7), intensity=0.9))
    set_camera(plotter, (3.8, 2.7, -4.3), (0.45, 0.45, 0.75), parallel_scale=2.2)
    plotter.screenshot(ASSET_DIR / "spin_axis_polar_coordinates_pyvista.png")
    plotter.close()


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    pv.global_theme.font.family = "arial"
    render_coordinate_system()
    render_polar_coordinates()
    render_forces_on_ball()
    render_spin_decomposition()
    render_spin_axis_polar_coordinates()


if __name__ == "__main__":
    main()
