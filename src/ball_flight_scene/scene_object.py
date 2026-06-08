from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class SceneObject(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def draw(self, fig, color: str | None = None, linewidth: int | None = None) -> None:
        pass

    @abstractmethod
    def get_3d_coordinates(self) -> NDArray[np.float64]:
        pass
