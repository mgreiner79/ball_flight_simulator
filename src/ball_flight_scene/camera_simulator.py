from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


class CameraSimulator:
    def __init__(
        self,
        f_mm: float,
        pixel_size_um: NDArray[np.float64],
        dimensions: NDArray[np.float64],
        C: NDArray[np.float64],
        rotation: NDArray[np.float64],
        length: float = 3,
    ):
        self.f_m = self._convert_mm_to_m(f_mm)
        self.dimensions = dimensions
        self.pixel_size_m = self._convert_um_to_m(pixel_size_um)
        self.C = C
        self.f_px = self._convert_f_to_pixels(self.f_m, self.pixel_size_m)
        self.rotation = rotation
        self.R = self._build_rotation_matrix(self.rotation)
        self.K = self._build_camera_matrix(self.f_px, self.dimensions)
        self.default_color = "orange"
        self.default_linewidth = 4
        self.length = length

    def _convert_f_to_pixels(self, focal_length_m, pixel_size_m):
        return focal_length_m / pixel_size_m

    def _convert_mm_to_m(self, mm_value):
        return mm_value / 1000

    def _convert_um_to_m(self, um_value):
        return um_value / 1e6

    def _build_rotation_matrix(self, rotation: NDArray[np.float64]):
        yaw_deg, pitch_deg, roll_deg = rotation
        r = R.from_euler("xyz", [yaw_deg, pitch_deg, roll_deg], degrees=True)
        return r.as_matrix()

    def _build_camera_matrix(self, f_px, dimensions):
        fx, fy = f_px
        cx = dimensions[0] / 2
        cy = dimensions[1] / 2

        return np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

    def set_rotation(self, yaw_deg, pitch_deg, roll_deg):
        self.rotation = np.array([yaw_deg, pitch_deg, roll_deg])
        self.R = self._build_rotation_matrix(self.rotation)

    def set_position(self, x, y, z):
        self.C = np.array([x, y, z])

    def offset(self, dx=0, dy=0, dz=0):
        self.C = np.array([self.C[0] + dx, self.C[1] + dy, self.C[2] + dz])

    def project(self, points_3d):
        N = points_3d.shape[0]
        points_3d_hom = np.hstack((points_3d, np.ones((N, 1))))

        t = -self.R @ self.C.reshape(-1, 1)
        extrinsic_matrix = np.hstack((self.R, t))
        P = self.K @ extrinsic_matrix

        points_2d_hom = (P @ points_3d_hom.T).T
        return points_2d_hom[:, :2] / points_2d_hom[:, 2][:, np.newaxis]

    def _build_edges(self):
        sensor_width = self.dimensions[0] * self.pixel_size_m[0]
        sensor_height = self.dimensions[1] * self.pixel_size_m[1]
        width = (sensor_width / self.f_m) * self.length
        height = (sensor_height / self.f_m) * self.length
        length = self.length

        def camera_to_world(coord):
            return self.R.T @ coord + self.C

        V0 = np.array([0, 0, 0])
        V1 = np.array([0, 0, length])
        V2 = np.array([width / 2, height / 2, length])
        V3 = np.array([-width / 2, height / 2, length])
        V4 = np.array([width / 2, -height / 2, length])
        V5 = np.array([-width / 2, -height / 2, length])

        V0, V1, V2, V3, V4, V5 = [camera_to_world(p) for p in [V0, V1, V2, V3, V4, V5]]

        return [
            (V0, V1),
            (V0, V2),
            (V0, V3),
            (V0, V4),
            (V0, V5),
            (V2, V3),
            (V4, V5),
            (V3, V5),
            (V2, V4),
        ]

    def draw(self, fig, color=None, linewidth=None):
        edges = self._build_edges()
        color = color or self.default_color
        linewidth = linewidth or self.default_linewidth

        for idx, edge in enumerate(edges):
            x_edge, y_edge, z_edge = zip(*edge)
            fig.add_trace(
                go.Scatter3d(
                    x=x_edge,
                    y=y_edge,
                    z=z_edge,
                    mode="lines",
                    line=dict(color=color, width=linewidth),
                    name="Camera" if idx == 0 else None,
                    showlegend=idx == 0,
                )
            )

    def render_projection(self, points, renderer="browser"):
        baseball_diameter_m = 0.074
        scaling_factor = 3
        fig = go.Figure()

        baseball_radius_m = baseball_diameter_m / 2
        cam_pos = self.C.flatten()
        distances = np.linalg.norm(points - cam_pos, axis=1)
        projections = self.project(points)
        apparent_size_px = 2 * self.f_px[0] * baseball_radius_m / distances
        marker_sizes = apparent_size_px * scaling_factor
        time = np.arange(len(points))
        sorted_indices = np.argsort(distances)[::-1]

        fig.add_trace(
            go.Scatter(
                x=projections[:, 0][sorted_indices],
                y=projections[:, 1][sorted_indices],
                mode="markers",
                marker=dict(
                    size=marker_sizes[sorted_indices],
                    color=time[sorted_indices],
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
        )

        fig.update_layout(
            xaxis=dict(title="X (pixels)", range=[0, self.dimensions[0]], fixedrange=True),
            yaxis=dict(title="Y (pixels)", range=[self.dimensions[1], 0], fixedrange=True),
            yaxis_scaleanchor="x",
        )
        fig.show(renderer=renderer)
