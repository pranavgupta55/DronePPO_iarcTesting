"""Idea 9, time-to-wall barrier. For each of the four walls, time to impact tw = distance / max(closing speed, eps). Take the smallest tw and
reward -k * exp(-tw / tau). Speed-independent: a fast drone far from a wall is safe, a slow drone very close is not penalised much.
Dense part: approach speed, time penalty, spin penalty above a deadband."""
import math

NAME = "time_to_wall_barrier"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"k": 1.0, "tau": 0.5, "eps": 1.0, "W": 1200.0, "H": 800.0, "approach_w": 1.0, "vel_ref": 150.0, "time_penalty": 0.1,
            "ang_vel_pen": 0.5, "deadband": 0.3}


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
        eps = p["eps"]
        # (distance to wall, closing speed toward it) for left, right, bottom, top
        walls = ((ctx.x, -ctx.vx), (p["W"] - ctx.x, ctx.vx), (ctx.y, -ctx.vy), (p["H"] - ctx.y, ctx.vy))
        tw = min(max(dist, 0.0) / max(v, eps) for dist, v in walls)
        r = -p["k"] * math.exp(-tw / p["tau"])
        r += p["approach_w"] * max(-1.0, min(1.0, ctx.approach_speed / p["vel_ref"])) - p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        return r
