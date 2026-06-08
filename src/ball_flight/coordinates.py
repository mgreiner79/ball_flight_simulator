"""Coordinate-system conventions used by the simulator.

The core physics package uses a camera-friendly right-handed coordinate system:

- X is right.
- Y is up.
- Z is forward, from release point toward home plate.

All distances are meters, time is seconds, velocity is meters per second, and
spin rate is revolutions per minute at the public API boundary.
"""

X_AXIS = "right"
Y_AXIS = "up"
Z_AXIS = "forward"
