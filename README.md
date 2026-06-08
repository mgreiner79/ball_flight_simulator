# Ball Flight Simulator

Reusable baseball flight simulation with optional scene visualization helpers.

The core package is `ball_flight`. It contains only the physics model and simple
data/export helpers, so it can be imported from other apps such as Blender.

Visualization and camera tools live in `ball_flight_scene`. Treat those modules,
the `examples/` folder, and the notebooks as inspection and validation tools.

## Core Usage

```python
from ball_flight import BallPathSimulator, PitchParameters

params = PitchParameters(
    v0=42,
    theta=0,
    phi=2,
    spin_rate=1200,
    spin_elevation=180,
    spin_azimuth=0,
    x0=-0.5,
    y0=1.6,
    z0=0,
)

trajectory = BallPathSimulator().calculate(params, t_max=5, dt=0.005)
positions = trajectory.positions
```

Install locally for use from other projects:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## Layout

- `src/ball_flight/`: reusable simulation package
- `src/ball_flight_scene/`: Plotly/PyVista-oriented inspection helpers
- `examples/`: scripts for simulation, scene viewing, and Blender-oriented export
- `data/`: sample detected ball-position data
- `tests/`: focused regression tests for the reusable simulator

## Physics Model

The simulator uses an idealized baseball-flight model:

- gravity
- quadratic drag with constant `C_d`
- Magnus lift based on effective spin perpendicular to the velocity vector
- constant air density
- constant spin vector during flight
- no wind, spin decay, seam orientation, or seam-shifted wake effects

Integration currently uses semi-implicit Euler. This is simple and stable enough
for exploratory use, but it is not a calibrated professional pitch-tracking
model.

For a detailed physics and code walkthrough, see
[Baseball Flight Physics Tutorial](docs/baseball_flight_physics_tutorial.md).
