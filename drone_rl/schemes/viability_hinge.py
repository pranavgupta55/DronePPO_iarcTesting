"""Idea 10, viability hinge (braking feasibility). For each wall, m_i = dist_i - max(closing speed, 0)^2 / (2*amax) is the margin left after
braking at amax; the drone is viable while min(m_i) >= 0. Reward -k * min(max(0, -min m) / scale_px, viol_cap). amax is set below the true
thrust limit (4000 by default) for safety. Dense part: approach speed, time penalty, spin penalty above a deadband."""

NAME = "viability_hinge"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"k": 1.0, "amax": 4000.0, "scale_px": 100.0, "viol_cap": 10.0, "approach_w": 1.0, "vel_ref": 150.0,
            "time_penalty": 0.1, "ang_vel_pen": 0.5, "deadband": 0.3}


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
        two_a = 2.0 * p["amax"]
        # (distance to wall, closing speed toward it) for left, right, bottom, top
        walls = ((ctx.x, -ctx.vx), (ctx.W - ctx.x, ctx.vx), (ctx.y, -ctx.vy), (ctx.H - ctx.y, ctx.vy))
        margin = min(max(dist, 0.0) - max(v, 0.0) ** 2 / two_a for dist, v in walls)
        violation = min(max(0.0, -margin) / p["scale_px"], p["viol_cap"])
        r = -p["k"] * violation
        r += p["approach_w"] * max(-1.0, min(1.0, ctx.approach_speed / p["vel_ref"])) - p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        return r
