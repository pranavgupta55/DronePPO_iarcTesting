"""Dense approach-speed reward: vel_w * clip(approach_speed / vel_ref, -1, 1) - time_penalty - spin penalty, plus an optional outward-nozzle term."""
import math

NAME = "approach_velocity"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"vel_w": 1.0, "vel_ref": 150.0, "time_penalty": 0.5, "ang_vel_pen": 0.5, "deadband": 0.3, "outward_w": 0.0, "outward_target": math.pi / 6}


def outward_term(angles_out, target):
    """ppo.py's outward reward in [-1, 1]: each nozzle scores 1 - |outward angle - target| / target, peaking at `target`."""
    return sum(max(-1.0, 1 - abs(a - target) / target) for a in angles_out) / 2


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
        r = p["vel_w"] * max(-1.0, min(1.0, ctx.approach_speed / p["vel_ref"])) - p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        if p["outward_w"]:
            r += p["outward_w"] * outward_term(ctx.angles_out, p["outward_target"])
        return r
