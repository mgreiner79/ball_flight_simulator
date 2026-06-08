from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from ..scene_object import SceneObject


class StrikeZone(SceneObject):
    def __init__(
        self,
        d_mound: float,
        z_mound: float,
        length: float = 0.44,
        depth: float = 0.216,
        y_min: float = 0.5,
        y_max: float = 1.1,
    ):
        super().__init__("Strike Zone")
        z_min = d_mound - z_mound
        z_max = z_min + depth
        x_min = -length / 2
        x_max = length / 2

        V0 = [x_min, y_min, z_min]
        V1 = [x_min, y_min, z_max]
        V2 = [x_min, y_max, z_max]
        V3 = [x_min, y_max, z_min]
        V4 = [x_max, y_min, z_min]
        V5 = [x_max, y_min, z_max]
        V6 = [x_max, y_max, z_max]
        V7 = [x_max, y_max, z_min]

        self.edges = [
            (V0, V1),
            (V1, V2),
            (V2, V3),
            (V3, V0),
            (V4, V5),
            (V5, V6),
            (V6, V7),
            (V7, V4),
            (V0, V4),
            (V1, V5),
            (V2, V6),
            (V3, V7),
        ]
        self.default_color = "red"
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
