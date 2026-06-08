"""Core baseball flight simulation.

This module intentionally contains no plotting, scene, camera, or Blender code.
It is the reusable physics layer: callers provide initial pitch parameters and
receive a time-aligned :class:`~ball_flight.models.Trajectory`.

Coordinate convention:

- X is horizontal side-to-side, positive to the right.
- Y is vertical, positive upward.
- Z is forward along the pitch direction.

Public inputs use SI units except for angles and spin:

- distances: meters
- velocity: meters per second
- time: seconds
- angles: degrees
- spin rate: revolutions per minute

The aerodynamic model is intentionally idealized. It includes gravity,
quadratic drag, and Magnus lift from the spin component perpendicular to the
instantaneous velocity vector. It does not include wind, spin decay, changing
air density, seam orientation, or seam-shifted wake effects.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .models import FloatArray, PitchParameters, Trajectory


_EPS = 1e-12


class BallPathSimulator:
    """Simulate a baseball trajectory with gravity, drag, and Magnus lift.

    The preferred API is :meth:`calculate`, which takes a
    :class:`PitchParameters` object and returns an immutable-ish
    :class:`Trajectory` result. The older stateful flow,
    :meth:`initialize_simulation` followed by :meth:`calculate_path`, is kept for
    compatibility with exploratory notebooks and examples.

    The integrator is semi-implicit Euler:

    1. compute forces and acceleration from the current state;
    2. update velocity from acceleration;
    3. update position using the new velocity.

    This is simple and stable enough for exploratory use, but it is still a
    first-order method. If this grows into a calibrated pitch-tracking model,
    the integration method should likely become RK4 or an adaptive ODE solver.
    """

    def __init__(
        self,
        name: str = "Ball Path",
        g: float = 9.81,
        rho: float = 1.225,
        C_d: float = 0.3,
        r: float = 0.0365,
        m: float = 0.145,
    ):
        """Create a simulator with physical constants.

        Parameters
        ----------
        name:
            Human-readable label used by legacy scene/plotting adapters.
        g:
            Gravitational acceleration in m/s^2.
        rho:
            Air density in kg/m^3. Use 0 to disable aerodynamic forces from air.
        C_d:
            Dimensionless drag coefficient. This implementation keeps it
            constant throughout the trajectory.
        r:
            Ball radius in meters. The default is close to a baseball radius.
        m:
            Ball mass in kilograms. The default is close to a baseball mass.
        """
        self.name = name
        self.g = g
        self.rho = rho
        self.C_d = C_d
        self.r = r
        self.A = np.pi * r**2
        self.m = m
        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)
        self.z = np.array([], dtype=np.float64)
        self.vx = np.array([], dtype=np.float64)
        self.vy = np.array([], dtype=np.float64)
        self.vz = np.array([], dtype=np.float64)
        self.time = np.array([], dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)
        self.C_l = np.array([], dtype=np.float64)

    def initialize_simulation(
        self,
        v0: float,
        theta: float,
        phi: float,
        spin_rate: float,
        spin_elevation: float,
        spin_azimuth: float,
        x0: float,
        y0: float,
        z0: float,
    ) -> None:
        """Set initial pitch conditions on the simulator instance.

        This method exists mainly for the legacy stateful workflow. New code
        should usually call :meth:`calculate` instead.

        Parameters
        ----------
        v0:
            Initial speed in m/s.
        theta:
            Vertical launch angle in degrees. Positive values point upward.
        phi:
            Horizontal launch angle in degrees. Positive values add rightward X
            velocity while most motion remains along positive Z.
        spin_rate:
            Total spin rate in revolutions per minute.
        spin_elevation:
            Spin-axis elevation angle in degrees in the local pitch frame.
        spin_azimuth:
            Spin-axis azimuth angle in degrees in the local pitch frame.
        x0, y0, z0:
            Initial position in meters.
        """
        self._validate_pitch_parameters(
            v0=v0,
            theta=theta,
            phi=phi,
            spin_rate=spin_rate,
            spin_elevation=spin_elevation,
            spin_azimuth=spin_azimuth,
            x0=x0,
            y0=y0,
            z0=z0,
        )
        self._validate_physical_constants()

        self.params = PitchParameters(
            v0=v0,
            theta=theta,
            phi=phi,
            spin_rate=spin_rate,
            spin_elevation=spin_elevation,
            spin_azimuth=spin_azimuth,
            x0=x0,
            y0=y0,
            z0=z0,
        )

        self.v0 = v0
        self.theta = np.radians(theta)
        self.phi = np.radians(phi)
        self.spin_rate = spin_rate
        self.spin_elevation = np.radians(spin_elevation)
        self.spin_azimuth = np.radians(spin_azimuth)
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.C_l = np.array([], dtype=np.float64)

        self._set_initial_velocity()
        self._init_positions()
        self._init_velocities()
        self._init_spin_vector()

    def calculate(
        self,
        params: PitchParameters,
        t_max: float = 5,
        dt: float = 0.01,
    ) -> Trajectory:
        """Run a complete simulation and return the trajectory.

        This is the recommended package API. It resets the simulator state from
        ``params``, integrates until ``t_max`` or ground impact, and returns a
        :class:`Trajectory` whose arrays are time-aligned.

        Parameters
        ----------
        params:
            Initial pitch conditions.
        t_max:
            Maximum simulation time in seconds.
        dt:
            Fixed integration time step in seconds.

        Returns
        -------
        Trajectory
            Time, position, velocity, acceleration, and aerodynamic diagnostic
            arrays. Each first dimension has the same length.
        """
        self.initialize_simulation(**asdict(params))
        self.calculate_path(t_max=t_max, dt=dt)
        return self.get_trajectory()

    def _validate_pitch_parameters(self, **params: float) -> None:
        """Validate the public pitch-input values before state initialization."""
        for name, value in params.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if params["v0"] <= 0:
            raise ValueError("v0 must be a positive finite number.")
        if params["spin_rate"] < 0:
            raise ValueError("spin_rate must be non-negative and finite.")

    def _validate_physical_constants(self) -> None:
        """Validate physical constants used in force calculations."""
        if not np.isfinite(self.m) or self.m <= 0:
            raise ValueError("Ball mass must be positive and finite.")
        if not np.isfinite(self.r) or self.r <= 0:
            raise ValueError("Ball radius must be positive and finite.")
        if not np.isfinite(self.rho) or self.rho < 0:
            raise ValueError("Air density must be non-negative and finite.")
        if not np.isfinite(self.C_d) or self.C_d < 0:
            raise ValueError("Drag coefficient must be non-negative and finite.")
        if not np.isfinite(self.g) or self.g < 0:
            raise ValueError("Gravity must be non-negative and finite.")

    def _validate_simulation_settings(self, t_max: float, dt: float) -> None:
        """Validate integration controls before allocating trajectory arrays."""
        if not np.isfinite(t_max) or t_max <= 0:
            raise ValueError("t_max must be a positive finite number.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number.")

    def _require_initialized(self) -> None:
        """Ensure the legacy stateful API is being called in a valid order."""
        if not hasattr(self, "x0") or not hasattr(self, "vx0"):
            raise ValueError("initialize_simulation() must be called before calculate_path().")

    def _set_initial_velocity(self) -> None:
        """Resolve initial speed and launch angles into X/Y/Z velocity.

        The velocity direction follows the module coordinate convention:
        positive Z is forward, positive Y is upward, and positive X is right.
        """
        self.vz0 = self.v0 * np.cos(self.theta) * np.cos(self.phi)
        self.vx0 = self.v0 * np.cos(self.theta) * np.sin(self.phi)
        self.vy0 = self.v0 * np.sin(self.theta)

    def _init_spin_vector(self) -> None:
        """Convert local spin-axis parameters into a world-coordinate vector.

        The public spin angles are easier to reason about in a local frame tied
        to the initial pitch direction. This method builds an orthonormal basis
        around the initial velocity and maps the local angular-velocity vector
        into the global X/Y/Z coordinate system.
        """
        spin_rate_rad_s = (self.spin_rate * 2 * np.pi) / 60

        velocity = np.array([self.vx0, self.vy0, self.vz0], dtype=np.float64)
        v_norm = np.linalg.norm(velocity)
        if v_norm == 0:
            raise ValueError("Initial velocity cannot be zero.")
        u_v = velocity / v_norm

        # Build two axes perpendicular to the initial velocity. Use world-up as
        # the first reference unless the pitch is too close to vertical.
        arbitrary = np.array([0, 1, 0], dtype=np.float64)
        u_perp1 = np.cross(u_v, arbitrary)
        if np.linalg.norm(u_perp1) == 0:
            arbitrary = np.array([1, 0, 0], dtype=np.float64)
            u_perp1 = np.cross(u_v, arbitrary)
        u_perp1 /= np.linalg.norm(u_perp1)
        u_perp2 = np.cross(u_v, u_perp1)

        # Local spin vector components are converted from rpm to radians/sec.
        omega_local = spin_rate_rad_s * np.array(
            [
                np.cos(self.spin_elevation) * np.cos(self.spin_azimuth),
                np.sin(self.spin_elevation),
                np.cos(self.spin_elevation) * np.sin(self.spin_azimuth),
            ],
            dtype=np.float64,
        )

        transformation_matrix = np.column_stack((u_perp1, u_perp2, u_v))
        self.omega = transformation_matrix @ omega_local

    def _init_positions(self) -> None:
        """Initialize legacy position arrays with the release point."""
        self.x = np.array([self.x0], dtype=np.float64)
        self.y = np.array([self.y0], dtype=np.float64)
        self.z = np.array([self.z0], dtype=np.float64)

    def _init_velocities(self) -> None:
        """Initialize legacy velocity arrays with the release velocity."""
        self.vx = np.array([self.vx0], dtype=np.float64)
        self.vy = np.array([self.vy0], dtype=np.float64)
        self.vz = np.array([self.vz0], dtype=np.float64)

    def get_lift_coefficient(self, S: float) -> float:
        """Return the dimensionless lift coefficient for spin parameter ``S``.

        ``S`` is the spin parameter ``|omega_perp| * r / |v|``, where
        ``omega_perp`` is the spin component perpendicular to the current
        velocity vector. This implementation uses the simple formula already
        present in the original simulator:

        ``C_l = 1 / (2 + sqrt(2 / S))``

        For zero or near-zero effective spin, lift is set to zero.
        """
        if S <= _EPS:
            return 0.0
        return 1 / (2 + np.sqrt(2 / S))

    def calculate_effective_spin(self, velocity: FloatArray) -> FloatArray:
        """Return the spin component that can generate Magnus lift.

        Only spin perpendicular to the direction of travel contributes to the
        Magnus force. Spin parallel to velocity is gyrospin; it may affect ball
        orientation, but this simplified model treats it as producing no lift.

        Parameters
        ----------
        velocity:
            Current velocity vector in m/s.

        Returns
        -------
        FloatArray
            Angular-velocity vector in rad/s after removing the component
            parallel to ``velocity``.
        """
        v_norm = np.linalg.norm(velocity)
        if v_norm <= _EPS:
            return np.zeros(3, dtype=np.float64)

        u_v = velocity / v_norm
        # Project omega onto the plane normal to velocity:
        # omega_perp = omega - (omega . u_v) u_v
        omega_perp = self.omega - np.dot(self.omega, u_v) * u_v
        if np.linalg.norm(omega_perp) <= _EPS:
            return np.zeros(3, dtype=np.float64)
        return omega_perp

    def calculate_lift_force(
        self,
        velocity: FloatArray,
        C_l: float,
        v: float,
        effective_spin: FloatArray,
    ) -> FloatArray:
        """Calculate the Magnus lift force vector in newtons.

        The direction is perpendicular to both effective spin and velocity:
        ``omega_perp x velocity``. The magnitude uses the standard dynamic
        pressure form:

        ``0.5 * rho * A * C_l * |v|^2``

        where ``A`` is the ball cross-sectional area.
        """
        lift_force_direction = np.cross(effective_spin, velocity)
        if np.linalg.norm(lift_force_direction) > _EPS:
            lift_force_direction /= np.linalg.norm(lift_force_direction)
        else:
            lift_force_direction = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        F_l_magnitude = 0.5 * self.rho * self.A * C_l * v**2
        return F_l_magnitude * lift_force_direction

    def calculate_drag_magnitude(self, v: float) -> float:
        """Return quadratic drag-force magnitude in newtons."""
        return 0.5 * self.rho * v**2 * self.C_d * self.A

    def calculate_spin_parameter(self, v: float, effective_spin: FloatArray) -> float:
        """Return dimensionless spin parameter based on effective spin.

        The model uses ``S = |omega_perp| * r / |v|``. Using effective spin here
        prevents near-gyrospin pitches from receiving unrealistically large
        lift coefficients.
        """
        if v <= _EPS:
            return 0.0
        return np.linalg.norm(effective_spin) * self.r / v

    def get_velocity_vector(self, i: int) -> FloatArray:
        """Return velocity at trajectory index ``i`` as ``[vx, vy, vz]``."""
        return np.array([self.vx[i], self.vy[i], self.vz[i]], dtype=np.float64)

    def get_velocity_magnitude(self, velocity_vector: FloatArray) -> float:
        """Return Euclidean speed from a velocity vector."""
        return np.linalg.norm(velocity_vector)

    def calculate_acceleration(
        self,
        i: int,
        v: float,
        F_d: float,
        lift_force: FloatArray,
    ) -> tuple[float, float, float]:
        """Calculate acceleration from drag, gravity, and lift.

        Drag acts opposite the current velocity direction. Gravity acts in
        negative Y. Lift is supplied as a fully resolved force vector.
        """
        ax = (-F_d * (self.vx[i] / v) + lift_force[0]) / self.m
        ay = -self.g + ((-F_d * (self.vy[i] / v) + lift_force[1]) / self.m)
        az = (-F_d * (self.vz[i] / v) + lift_force[2]) / self.m
        return ax, ay, az

    def set_acceleration(self, i: int, acceleration: tuple[float, float, float]) -> None:
        """Store acceleration components at trajectory index ``i``."""
        ax, ay, az = acceleration
        self.ax[i] = ax
        self.ay[i] = ay
        self.az[i] = az

    def calculate_next_velocity(
        self,
        velocity: FloatArray,
        acceleration: tuple[float, float, float],
        dt: float,
    ) -> tuple[float, float, float]:
        """Advance velocity one fixed time step using Euler integration."""
        vx0, vy0, vz0 = velocity
        ax, ay, az = acceleration
        return vx0 + ax * dt, vy0 + ay * dt, vz0 + az * dt

    def set_velocity(self, i: int, velocity: tuple[float, float, float]) -> None:
        """Store velocity components at trajectory index ``i``."""
        vx, vy, vz = velocity
        self.vx[i] = vx
        self.vy[i] = vy
        self.vz[i] = vz

    def calculate_next_position(
        self,
        position: tuple[float, float, float],
        velocity: FloatArray,
        dt: float,
    ) -> tuple[float, float, float]:
        """Advance position one fixed time step using the supplied velocity."""
        x, y, z = position
        vx, vy, vz = velocity
        return x + vx * dt, y + vy * dt, z + vz * dt

    def set_position(self, i: int, position: tuple[float, float, float]) -> None:
        """Store position components at trajectory index ``i``."""
        self.x[i] = position[0]
        self.y[i] = position[1]
        self.z[i] = position[2]

    def get_position(self, i: int) -> tuple[float, float, float]:
        """Return position at trajectory index ``i`` as ``(x, y, z)``."""
        return self.x[i], self.y[i], self.z[i]

    def calculate_path(self, t_max: float = 5, dt: float = 0.01) -> None:
        """Integrate the pitch path into the simulator's state arrays.

        This is the legacy stateful integration API. It assumes
        :meth:`initialize_simulation` has already populated release position,
        release velocity, and spin vector. Prefer :meth:`calculate` for normal
        use.

        The method stops when either:

        - the fixed time grid reaches ``t_max``; or
        - the ball crosses the ground plane, ``y = 0``.

        If ground contact occurs between two time samples, the final stored
        sample is linearly interpolated to ``y = 0`` so callers get an explicit
        impact point instead of the last above-ground point.
        """
        self._require_initialized()
        self._validate_simulation_settings(t_max, dt)
        self._validate_physical_constants()

        # Preserve the historical time-grid behavior: include samples from
        # 0 <= t < t_max at fixed spacing dt.
        self.time = np.arange(0, t_max, dt)
        num_steps = len(self.time)
        self.x = np.zeros(num_steps)
        self.y = np.zeros(num_steps)
        self.z = np.zeros(num_steps)
        self.vx = np.zeros(num_steps)
        self.vy = np.zeros(num_steps)
        self.vz = np.zeros(num_steps)

        self.F_d = np.zeros(num_steps)
        self.S = np.zeros(num_steps)
        self.v = np.zeros(num_steps)
        self.lift_force = np.zeros((num_steps, 3))
        self.ax = np.zeros(num_steps)
        self.ay = np.zeros(num_steps)
        self.az = np.zeros(num_steps)
        self.C_l = np.zeros(num_steps)

        self.x[0], self.y[0], self.z[0] = self.x0, self.y0, self.z0
        self.vx[0], self.vy[0], self.vz[0] = self.vx0, self.vy0, self.vz0

        for i in range(0, num_steps):
            # Diagnostics and acceleration are computed from the current state.
            position = self.get_position(i)
            velocity = self.get_velocity_vector(i)
            self.v[i] = self.get_velocity_magnitude(velocity)
            v = self.v[i]

            if v == 0:
                continue

            self.F_d[i] = self.calculate_drag_magnitude(v)
            F_d = self.F_d[i]

            # Effective spin is recalculated each step because velocity changes
            # under gravity, drag, and lift while this simplified model keeps
            # the world-coordinate spin vector fixed.
            effective_spin = self.calculate_effective_spin(velocity)
            self.S[i] = self.calculate_spin_parameter(v, effective_spin)
            S = self.S[i]
            self.C_l[i] = self.get_lift_coefficient(S)
            self.lift_force[i] = self.calculate_lift_force(velocity, self.C_l[i], v, effective_spin)
            lift_force = self.lift_force[i]

            acceleration = self.calculate_acceleration(i, v, F_d, lift_force)
            self.set_acceleration(i, acceleration)

            if i == num_steps - 1:
                break

            next_velocity = self.calculate_next_velocity(velocity, acceleration, dt)

            # Semi-implicit Euler: position advances with the updated velocity.
            next_position = self.calculate_next_position(
                position,
                np.asarray(next_velocity, dtype=np.float64),
                dt,
            )

            if next_position[1] <= 0:
                # Replace the first below-ground step with a linearly
                # interpolated impact sample on the ground plane.
                impact_position, impact_velocity, impact_time = self._interpolate_ground_impact(
                    position,
                    velocity,
                    next_position,
                    np.asarray(next_velocity, dtype=np.float64),
                    self.time[i],
                    dt,
                )
                self.time[i + 1] = impact_time
                self.set_position(i + 1, impact_position)
                self.set_velocity(i + 1, tuple(impact_velocity))
                self._set_diagnostics_for_state(i + 1)
                self._truncate(i + 2)
                break

            self.set_velocity(i + 1, next_velocity)
            self.set_position(i + 1, next_position)

    def _interpolate_ground_impact(
        self,
        position: tuple[float, float, float],
        velocity: FloatArray,
        next_position: tuple[float, float, float],
        next_velocity: FloatArray,
        time: float,
        dt: float,
    ) -> tuple[tuple[float, float, float], FloatArray, float]:
        """Interpolate the state where a step crosses the ground plane.

        The integrator advances in finite time steps, so a proposed next
        position may have ``y < 0``. This method linearly interpolates between
        the previous state and proposed next state to estimate the impact state
        at ``y = 0``.

        The interpolation is simple rather than ballistic. It is intended to
        keep output arrays useful and non-negative, not to provide an exact
        collision solve.
        """
        position_array = np.asarray(position, dtype=np.float64)
        next_position_array = np.asarray(next_position, dtype=np.float64)
        y_delta = next_position_array[1] - position_array[1]
        fraction = 1.0 if abs(y_delta) <= _EPS else (0.0 - position_array[1]) / y_delta
        fraction = float(np.clip(fraction, 0.0, 1.0))

        impact_position = position_array + fraction * (next_position_array - position_array)
        impact_position[1] = 0.0
        impact_velocity = velocity + fraction * (next_velocity - velocity)
        impact_time = time + fraction * dt
        return tuple(impact_position), impact_velocity, impact_time

    def _set_diagnostics_for_state(self, i: int) -> None:
        """Populate force and acceleration diagnostics for an existing state.

        This is used for the interpolated ground-impact sample. Since impact is
        inserted after the normal integration step, its diagnostic arrays need
        to be recomputed before truncating the trajectory.
        """
        velocity = self.get_velocity_vector(i)
        self.v[i] = self.get_velocity_magnitude(velocity)
        v = self.v[i]
        if v <= _EPS:
            return

        self.F_d[i] = self.calculate_drag_magnitude(v)
        effective_spin = self.calculate_effective_spin(velocity)
        self.S[i] = self.calculate_spin_parameter(v, effective_spin)
        self.C_l[i] = self.get_lift_coefficient(self.S[i])
        self.lift_force[i] = self.calculate_lift_force(velocity, self.C_l[i], v, effective_spin)
        self.set_acceleration(
            i,
            self.calculate_acceleration(i, v, self.F_d[i], self.lift_force[i]),
        )

    def _truncate(self, stop: int) -> None:
        """Trim all time-aligned arrays to the first ``stop`` samples."""
        self.x = self.x[:stop]
        self.y = self.y[:stop]
        self.z = self.z[:stop]
        self.vx = self.vx[:stop]
        self.vy = self.vy[:stop]
        self.vz = self.vz[:stop]
        self.time = self.time[:stop]
        self.F_d = self.F_d[:stop]
        self.S = self.S[:stop]
        self.v = self.v[:stop]
        self.lift_force = self.lift_force[:stop]
        self.ax = self.ax[:stop]
        self.ay = self.ay[:stop]
        self.az = self.az[:stop]
        self.C_l = self.C_l[:stop]

    def get_path_array(self) -> FloatArray:
        """Return ``[x, y, z, time]`` columns for legacy consumers."""
        return np.column_stack((self.x, self.y, self.z, self.time))

    def get_3d_coordinates(self) -> FloatArray:
        """Return positions as an ``(N, 3)`` array."""
        return np.array([self.x, self.y, self.z]).T

    def get_path(self) -> FloatArray:
        """Return positions as a ``(3, N)`` array for legacy path drawing."""
        return np.array([self.x, self.y, self.z])

    def get_spin_vector(self) -> FloatArray:
        """Return a two-point unit vector showing initial spin direction.

        The returned shape is ``(3, 2)`` so legacy path plotting can draw it as
        a short line segment starting at the release point.
        """
        if np.linalg.norm(self.omega) != 0:
            omega_norm = np.linalg.norm(self.omega)
            ox = [self.x0, self.x0 + self.omega[0] / omega_norm]
            oy = [self.y0, self.y0 + self.omega[1] / omega_norm]
            oz = [self.z0, self.z0 + self.omega[2] / omega_norm]
            return np.array([ox, oy, oz])
        return np.array([[self.x0, self.x0], [self.y0, self.y0], [self.z0, self.z0]])

    def get_trajectory(self) -> Trajectory:
        """Build a time-aligned :class:`Trajectory` from current state arrays."""
        return Trajectory(
            time=self.time.copy(),
            positions=np.column_stack((self.x, self.y, self.z)),
            velocities=np.column_stack((self.vx, self.vy, self.vz)),
            acceleration=np.column_stack((self.ax, self.ay, self.az)),
            drag_force=self.F_d.copy(),
            spin_parameter=self.S.copy(),
            lift_force=self.lift_force.copy(),
            lift_coefficient=self.C_l.copy(),
        )
