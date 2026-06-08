from ball_flight import BallPathSimulator, PitchParameters


def main() -> None:
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
    print(trajectory.positions.shape)
    print(trajectory.positions[-1])


if __name__ == "__main__":
    main()
