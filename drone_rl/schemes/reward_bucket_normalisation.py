"""Scale-invariant sum of four terms: shaping (potential), time penalty, spin penalty, wall proximity. Each raw term r_j is divided by its running
RMS (EMA of r_j^2, kept across episodes), clipped to [-1, 1], and weighted by w_j. Signs are kept, so the term weights set the balance."""
import math

NAME = "reward_bucket_normalisation"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3, "wall_ref": 60.0,
            "w_shape": 1.0, "w_time": 0.2, "w_spin": 0.3, "w_wall": 0.3, "ema": 1e-3, "floor": 1e-2}


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.phi = 0.0
        self.ms = [1.0, 1.0, 1.0, 1.0]  # running mean squares of the four raw terms

    def reset(self, env):
        self.phi = -env.time_to_go()

    def on_target_spawn(self, env):
        self.phi = -env.time_to_go()

    def set_progress(self, p):
        pass

    def extra_obs(self, env):
        return []

    def step(self, ctx):
        p = self.p
        phi = 0.0 if ctx.reached else -ctx.time_to_go
        raw = (p["shaping_w"] * (phi - self.phi),
               -p["time_penalty"],
               -p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"]),
               -max(0.0, 1.0 - ctx.dist_to_wall / p["wall_ref"]))
        self.phi = phi
        weights = (p["w_shape"], p["w_time"], p["w_spin"], p["w_wall"])
        total = 0.0
        for j in range(4):
            r = raw[j]
            self.ms[j] += p["ema"] * (r * r - self.ms[j])
            scale = max(math.sqrt(self.ms[j]), p["floor"])
            total += weights[j] * max(-1.0, min(1.0, r / scale))
        return total
