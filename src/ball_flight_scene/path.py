from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from .scene_object import SceneObject


class Path3d(SceneObject):
    def __init__(
        self,
        coords: NDArray[np.float64],
        name: str,
        default_color: str = "blue",
        default_linewidth: int = 3,
    ):
        super().__init__(name)
        self.coords = coords
        self.default_color = default_color
        self.default_linewidth = default_linewidth

    def draw(self, fig: go.Figure, color: str | None = None, linewidth: int | None = None) -> None:
        fig.add_trace(
            go.Scatter3d(
                x=self.coords[0],
                y=self.coords[1],
                z=self.coords[2],
                mode="lines",
                line=dict(
                    color=color or self.default_color,
                    width=linewidth or self.default_linewidth,
                ),
                name=self.name,
            )
        )

    def get_3d_coordinates(self) -> NDArray[np.float64]:
        return np.array(self.coords).T
