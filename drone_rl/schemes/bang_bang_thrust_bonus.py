"""Idea 8, bang-bang thrust bonus. bb = (|F1 - 0.5| + |F2 - 0.5|) * 2 in [0, 2]; reward bb_k * bb while speed > v_thresh (moving only).
Dense part: approach speed, time penalty, spin penalty above a deadband."""

NAME = "bang_bang_thrust_bonus"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"bb_k": 0.01, "v_thresh": 100.0, "approach_w": 1.0, "vel_ref": 150.0, "time_penalty": 0.1, "ang_vel_pen": 0.5, "deadband": 0.3}


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
        r = p["approach_w"] * max(-1.0, min(1.0, ctx.approach_speed / p["vel_ref"])) - p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        if ctx.speed > p["v_thresh"]:
            bb = (abs(ctx.forces[0] - 0.5) + abs(ctx.forces[1] - 0.5)) * 2.0
            r += p["bb_k"] * bb
        return r
