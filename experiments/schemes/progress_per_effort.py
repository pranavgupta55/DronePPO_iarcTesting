"""Idea 4, progress per unit of thrust. ratio = (d_prev - d) / (dt * (eps + F1 + F2) * speed_ref), clipped to +-clip, reward = k * ratio.
Rewards closing distance cheaply (bang-bang bursts, little wasted thrust). Plus time penalty and spin penalty above a deadband."""

NAME = "progress_per_effort"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"k": 0.1, "eps": 0.5, "speed_ref": 100.0, "clip": 5.0, "dt": 0.02, "time_penalty": 0.1, "ang_vel_pen": 0.5, "deadband": 0.3}


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}

    def reset(self, env):
        pass

    def set_progress(self, p):
        pass

    def extra_obs(self, env):
        return []

    def step(self, ctx):
        p = self.p
        effort = p["eps"] + ctx.forces[0] + ctx.forces[1]
        ratio = (ctx.prev_dist - ctx.dist) / (p["dt"] * effort * p["speed_ref"])
        r = p["k"] * max(-p["clip"], min(p["clip"], ratio))
        r -= p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        return r
