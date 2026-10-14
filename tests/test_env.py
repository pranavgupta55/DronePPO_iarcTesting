"""The environment: reset/step shapes, per-run world size, strict reach rule, crash on leaving the world."""
import numpy as np

from drone_rl.env import DroneEnv, make_spec, load_scheme


def make_env(**spec_overrides):
    spec = make_spec({"scheme": "time_shaping", "params": {}, **spec_overrides})
    module, scheme = load_scheme(spec["scheme"], spec["params"])
    return DroneEnv(spec, module, scheme, max_steps=200)


def test_reset_and_step_shapes():
    env = make_env()
    obs = env.reset()
    assert obs.shape == (17,) and np.isfinite(obs).all()
    obs, reward, terminated, truncated = env.step(np.zeros(4))
    assert obs.shape == (17,) and np.isfinite(reward)
    assert not terminated and not truncated


def test_world_size_is_per_run():
    small, big = make_env(world=[1200, 800]), make_env(world=[2400, 1600])
    assert (small.W, small.H) == (1200, 800) and (big.W, big.H) == (2400, 1600)
    small.reset()
    big.reset()
    assert abs(small.drone.shell.center[0] - 600) < 1 and abs(big.drone.shell.center[0] - 1200) < 1
    for _ in range(50):  # targets stay inside the world
        small.spawn_target()
        assert 0 <= small.target[0] <= 1200 and 0 <= small.target[1] <= 800


def test_leaving_the_world_is_a_crash():
    env = make_env(world=[400, 300])
    env.reset()
    env.drone.shell.set_physical_state(np.array([5.0, 150.0]), 0.0, np.array([-800.0, 0.0], dtype=np.float32), 0.0)
    terminated = False
    for _ in range(20):
        _, _, terminated, _ = env.step(np.zeros(4))
        if terminated:
            break
    assert terminated


def test_registry_required_fields():
    import pytest
    from drone_rl import registry
    with pytest.raises(ValueError):
        registry.check_run_spec({"name": "x", "blurb": "only one line", "group": "g", "tags": ["t"]})
    registry.check_run_spec({"name": "x", "blurb": "short name\nwhat it tests", "group": "g", "tags": ["t"]})
