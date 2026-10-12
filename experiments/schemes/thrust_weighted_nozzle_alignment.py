"""Time-optimal shaping plus thrust-weighted nozzle alignment: align_w * sum_i F_i * cos(outward_i - target). A nozzle with no thrust costs nothing."""
import math

from schemes.time_shaping import Scheme as TimeShaping

NAME = "thrust_weighted_nozzle_alignment"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

BASE = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3}
DEFAULTS = {**BASE, "align_w": 0.5, "outward_target": math.pi / 6}


def alignment(forces, angles_out, target):
    """sum_i F_i * cos(outward_i - target): in [-2, 2], peaks when full thrust is at the target outward angle."""
    return sum(f * math.cos(a - target) for f, a in zip(forces, angles_out))


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__({k: self.q[k] for k in BASE})

    def step(self, ctx):
        q = self.q
        return super().step(ctx) + q["align_w"] * alignment(ctx.forces, ctx.angles_out, q["outward_target"])
