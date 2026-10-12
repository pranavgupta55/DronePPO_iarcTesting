"""Validates a reward scheme: python3 check_scheme.py <scheme_name> [params_json]"""
import json
import math
import random
import sys

import numpy as np

from core.config import OBS_BASE_DIM, ACT_DIM
from core.env import DroneEnv, make_spec, load_scheme, HOVER_THRUST


def main():
    name = sys.argv[1]
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    spec = make_spec({"scheme": name, "params": params, "seed": 0})
    module, scheme = load_scheme(name, params)
    random.seed(0)
    rng = np.random.default_rng(0)
    env = DroneEnv(spec, module, scheme)
    n_extra = int(module.EXTRA_OBS)
    obs_dim = OBS_BASE_DIM + n_extra
    rewards, resets = [], 0

    def check_obs(obs):
        assert obs.shape == (obs_dim,), f"obs has {obs.shape}, expected ({obs_dim},)"
        assert np.isfinite(obs).all(), f"non-finite observation {obs}"
        if n_extra:
            extra = scheme.extra_obs(env)
            assert len(extra) == n_extra, f"extra_obs returned {len(extra)} values, EXTRA_OBS={n_extra}"
            assert all(math.isfinite(e) for e in extra), f"non-finite extra_obs {extra}"

    for progress in (0.0, 0.5, 1.0):
        env.set_ramp(progress)
        for _ in range(3):  # a few episodes at each progress point
            check_obs(env.reset())
            resets += 1
            for _ in range(200):
                obs, r, term, trunc = env.step(rng.normal(0, 1, ACT_DIM))
                check_obs(obs)
                assert math.isfinite(r), f"non-finite reward {r}"
                rewards.append(r)
                if term or trunc:
                    check_obs(env.reset())
                    resets += 1

    # Forced reach: drop the drone, still, on the target with loose reach limits
    env.set_ramp(0.0)
    env.reset()
    env.drone.shell.set_physical_state(env.target, 0.0, np.zeros(2), 0.0)
    env.drone.thruster_forces = [HOVER_THRUST, HOVER_THRUST]
    env.drone.previous_thruster_forces = [HOVER_THRUST, HOVER_THRUST]
    reach_r, before = None, env.reaches
    for _ in range(5):
        obs, r, term, trunc = env.step(np.zeros(ACT_DIM))
        check_obs(obs)
        if env.reaches > before:
            reach_r = r
            break
    print(f"scheme {name} params {params}: EXTRA_OBS={n_extra}, obs_dim={obs_dim}, {len(rewards)} random steps, {resets} resets")
    print(f"reward/step mean {np.mean(rewards):.3f} min {np.min(rewards):.3f} max {np.max(rewards):.3f} std {np.std(rewards):.3f}")
    print("forced reach:", f"reward {reach_r:.1f} (includes env reach_reward {spec['reach_reward']:.0f})" if reach_r is not None else "not triggered")
    print("OK")


if __name__ == "__main__":
    main()
