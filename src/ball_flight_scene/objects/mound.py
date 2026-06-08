from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from ..scene_object import SceneObject


class Mound(SceneObject):
    def __init__(
        self,
        x_center: float,
        z_center: float,
        y_center: float = 0,
        length: float = 0.44,
        depth: float = 0.15,
        height: float = 0,
    ):
        super().__init__("Mound")
        self.x_min = x_center - length / 2
        self.x_max = x_center + length / 2
        self.z_min = z_center - depth / 2
        self.z_max = z_center + depth / 2
        self.y_min = y_center - height / 2

        V0 = [self.x_min, self.y_min, self.z_min]
        V1 = [self.x_min, self.y_min, self.z_max]
        V2 = [self.x_max, self.y_min, self.z_max]
        V3 = [self.x_max, self.y_min, self.z_min]

        self.edges = [(V0, V1), (V1, V2), (V2, V3), (V3, V0)]
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
