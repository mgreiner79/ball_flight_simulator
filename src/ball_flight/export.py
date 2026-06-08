from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import Trajectory


def trajectory_to_records(trajectory: Trajectory) -> list[dict[str, float]]:
    return [
        {"time": float(t), "x": float(x), "y": float(y), "z": float(z)}
        for t, (x, y, z) in zip(trajectory.time, trajectory.positions)
    ]


def trajectory_to_json(trajectory: Trajectory, path: str | Path) -> None:
    payload: dict[str, Any] = {"positions": trajectory_to_records(trajectory)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def trajectory_to_csv(trajectory: Trajectory, path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["time", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(trajectory_to_records(trajectory))
