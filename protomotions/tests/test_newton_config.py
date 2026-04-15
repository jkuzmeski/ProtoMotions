from protomotions.simulator.newton.config import NewtonSimulatorConfig, NewtonSimParams


def test_newton_viser_viewer_port_defaults_to_8097():
    config = NewtonSimulatorConfig(
        headless=True,
        num_envs=1,
        sim=NewtonSimParams(),
        experiment_name="test-newton-config",
    )
    assert config.viewer_port == 8097
    assert config.viewer_max_worlds == 16
