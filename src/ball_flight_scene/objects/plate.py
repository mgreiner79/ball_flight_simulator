from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from ..scene_object import SceneObject


class Plate(SceneObject):
    def __init__(self, d_mound: float, x_mound: float):
        super().__init__("Home Plate")
        plate_width = 0.43
        plate_depth = 0.216
        x_plate_front = d_mound - x_mound
        x_plate_back = x_plate_front + plate_depth
        y_plate = 0
        z_plate_min = -plate_width / 2
        z_plate_max = plate_width / 2

        P0 = [x_plate_front, y_plate, z_plate_min]
        P1 = [x_plate_front, y_plate, z_plate_max]
        P2 = [x_plate_back, y_plate, z_plate_max]
        P3 = [x_plate_back + plate_depth, y_plate, 0]
        P4 = [x_plate_back, y_plate, z_plate_min]

        self.edges = [(P0, P1), (P1, P2), (P2, P3), (P3, P4), (P4, P0)]
        self.default_color = "white"
        self.default_linewidth = 4

    def draw(self, fig: go.Figure, color: str | None = None, linewidth: int | None = None) -> None:
        color = color or self.default_color
        linewidth = linewidth or self.default_linewidth
        for idx, edge in enumerate(self.edges):
            x_edge, y_edge, z_edge = zip(*edge)
            fig.add_trace(
                go.Scatter3d(
                    x=x_edge,
                    y=y_edge,
                    z=z_edge,
                    mode="lines",
                    line=dict(color=color, width=linewidth),
                    name=self.name if idx == 0 else None,
                    showlegend=idx == 0,
                )
            )

    def get_3d_coordinates(self) -> NDArray[np.float64]:
        coords = []
        for edge in self.edges:
            coords.extend(edge)
        return np.array(coords)
