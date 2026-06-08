# Review: `src/ball_flight/simulator.py`

## Findings

1. High: `lift_coefficient` is not time-aligned with the trajectory result.

   `get_lift_coefficient()` appends to `self.C_l` only when `S != 0` (`simulator.py:130-135`), but `get_trajectory()` exposes it as if it were another time series (`simulator.py:303-314`). With a normal spinning pitch I observed `len(time) == 96` and `len(lift_coefficient) == 97`; with zero spin I observed `len(time) == 59` and `len(lift_coefficient) == 0`. This makes `Trajectory.lift_coefficient` unreliable for analysis, plotting, export, or frame-by-frame comparison.

   Suggested fix: allocate `self.C_l = np.zeros(num_steps)` in `calculate_path()` and assign `self.C_l[i] = C_l` instead of mutating a list from inside `get_lift_coefficient()`. Keep pure calculations side-effect free.

2. High: Magnus lift magnitude incorrectly uses total spin, not transverse/effective spin.

   `calculate_spin_parameter()` uses `np.linalg.norm(self.omega)` (`simulator.py:151-152`), while `calculate_lift_force()` normalizes `np.cross(self.omega, velocity)` and then applies a lift magnitude based on that total-spin `S` (`simulator.py:137-146`). This means a pitch with mostly gyrospin but a tiny transverse component can receive nearly full lift magnitude because the direction is normalized and the magnitude ignores spin-axis efficiency. If the spin is exactly parallel to velocity the force becomes zero, but nearly parallel spin gets a discontinuously large lift.

   Suggested fix: compute the spin component perpendicular to velocity, e.g. `omega_perp = omega - dot(omega, u_v) * u_v`, use `norm(omega_perp)` in `S`, and scale lift from that effective spin. Also add tests for pure gyrospin and near-gyrospin.

3. Medium: `time` can have a different length than the state arrays when `t_max / dt` is not an integer.

   `num_steps = int(t_max / dt)` (`simulator.py:213`) truncates, while `self.time = np.arange(0, t_max, dt)` (`simulator.py:214`) may produce more samples. For example, `t_max=1, dt=0.3` gives `len(time) == 4` but `len(x) == 3`, which will break `get_path_array()` and `get_trajectory()`.

   Suggested fix: derive time from `num_steps`, e.g. `self.time = np.arange(num_steps) * dt`, or compute `num_steps = len(np.arange(...))` and allocate arrays from that length.

4. Medium: invalid integration settings are not validated.

   `dt=0` raises a raw `ZeroDivisionError` at `simulator.py:213`; negative `dt` or nonpositive `t_max` will lead to invalid or empty arrays and later indexing errors at `simulator.py:231-232`. These are API-facing inputs now that the simulator is reusable.

   Suggested fix: validate `dt > 0`, `t_max > 0`, finite numeric parameters, and positive physical constants. Raise `ValueError` with a clear message.

5. Medium: ground contact handling drops the first below-ground point and does not estimate impact.

   The ground check happens after calculating and storing the next state (`simulator.py:253-267`) but tests `self.y[i] < 0`, then `_truncate(i)` excludes the current sample (`simulator.py:269-283`). This keeps only the last above-ground state and never reports an impact point at `y=0`. It also does unnecessary force/integration work for a sample already below ground.

   Suggested fix: check `next_position[1] <= 0` before committing the next point, and optionally interpolate between the previous and next state to append a final impact sample at `y=0`.

6. Medium: the integrator is first-order explicit Euler and updates position with the old velocity.

   Velocity is advanced using acceleration (`simulator.py:253-257`), but position is advanced with the previous velocity (`simulator.py:259-263`). This is simple, but it is low accuracy and visibly timestep-dependent for drag/lift problems. A 5 ms step may be acceptable for demos, but the reusable package should make the numerical method explicit and test convergence.

   Suggested fix: consider semi-implicit Euler at minimum, or an RK4 / `solve_ivp` style derivative function. Add a convergence test comparing smaller `dt` results against coarser `dt` results for plate-crossing position and speed.

7. Low: `calculate()` depends on `params.__dict__`.

   `calculate()` unpacks a dataclass via `params.__dict__` (`simulator.py:82`). This works now, but it is brittle if the dataclass gets derived fields or slots later.

   Suggested fix: use `dataclasses.asdict(params)` or pass the fields explicitly.

8. Low: several methods mutate public state in ways that make ordering assumptions easy to violate.

   Methods like `calculate_path()` assume `initialize_simulation()` has already created `x0`, `vx0`, and `omega` (`simulator.py:212-232`). Calling `calculate_path()` on a fresh object gives an attribute error. This is survivable for legacy compatibility, but less friendly for a reusable library.

   Suggested fix: prefer `calculate(params, ...)` as the primary API, mark the stateful API as legacy/internal, and have `calculate_path()` guard against missing initialization.

## Physics Assessment

The model is a reasonable first baseball-flight approximation: gravity, quadratic drag, and a Magnus-style lift term are all present. The defaults for baseball mass/radius are plausible, and the coordinate convention is clear enough after the repo split.

The largest physics issue is spin effectiveness. A baseball's lift comes from the spin component perpendicular to the flight direction, while the current code computes lift coefficient from total spin. That will over-predict movement for gyro-heavy pitches and make spin-axis experiments misleading.

The second largest physics limitation is using fixed `C_d` and a simple lift coefficient formula. Constant drag and a Karman-Sears-style lift coefficient can be useful for exploration, but real baseball aerodynamics depend on speed, seam orientation, Reynolds number, spin parameter, and pitch-specific seam effects. That is fine if documented as an idealized model; it should not be treated as calibrated pitch-tracking physics yet.

The code also assumes no wind, no spin decay, constant air density, and no seam-shifted wake effects. Those are acceptable omissions for a reusable core if the package names them as model assumptions.

## Recommended Next Steps

1. Fix trajectory shape invariants: every `Trajectory` array should either be length `N` or clearly documented otherwise.
2. Replace total-spin Magnus scaling with effective transverse spin.
3. Add input validation for `dt`, `t_max`, and physical constants.
4. Change integration to semi-implicit Euler or RK4, then add convergence tests.
5. Add model documentation explaining coordinate system, angle meanings, units, and known physics simplifications.

## Verification Performed

- Read `src/ball_flight/simulator.py`, `src/ball_flight/models.py`, and `tests/test_ball_flight.py`.
- Ran the existing unit tests before review; they pass.
- Ran targeted smoke checks for `lift_coefficient` length, zero-spin output, invalid `dt`, and non-integer `t_max / dt`.
