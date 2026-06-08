"""Run inside Blender after generating data/example_trajectory.json.

This script intentionally keeps Blender-specific code outside the core package.
"""

import json
from pathlib import Path

import bpy


def create_curve_from_trajectory(path: str | Path, curve_name: str = "Pitch Trajectory") -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    positions = payload["positions"]

    curve = bpy.data.curves.new(curve_name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2

    spline = curve.splines.new("POLY")
    spline.points.add(len(positions) - 1)
    for point, row in zip(spline.points, positions):
        point.co = (row["x"], row["z"], row["y"], 1)

    obj = bpy.data.objects.new(curve_name, curve)
    bpy.context.collection.objects.link(obj)


create_curve_from_trajectory("data/example_trajectory.json")
