"""Every reward scheme loads in both world sizes and returns finite rewards and observations."""
import pathlib

import numpy as np
import pytest

from drone_rl.env import DroneEnv, make_spec, load_scheme

NAMES = sorted(p.stem for p in (pathlib.Path(__file__).resolve().parent.parent / "drone_rl" / "schemes").glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("world", [[1200, 800], [2400, 1600]])
def test_scheme_runs(name, world):
    spec = make_spec({"scheme": name, "params": {}, "world": world})
    module, scheme = load_scheme(name, {})
    env = DroneEnv(spec, module, scheme, max_steps=60)
    rng = np.random.default_rng(0)
    obs = env.reset()
    assert obs.shape == (17 + int(module.EXTRA_OBS),)
    for _ in range(60):
        obs, reward, terminated, truncated = env.step(rng.normal(0, 0.1, 4))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            env.reset()
