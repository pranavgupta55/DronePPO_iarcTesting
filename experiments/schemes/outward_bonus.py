"""time_shaping plus ppo.py's outward-nozzle reward (peaked at outward_target) weighted by outward_w."""
import math

from schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS
from schemes.approach_velocity import outward_term

NAME = "outward_bonus"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "outward_w": 1.0, "outward_target": math.pi / 6}


class Scheme(TimeShaping):
    def __init__(self, params):
        super().__init__({k: v for k, v in params.items() if k in TS_DEFAULTS})
        self.q = {**DEFAULTS, **params}

    def step(self, ctx):
        return super().step(ctx) + self.q["outward_w"] * outward_term(ctx.angles_out, self.q["outward_target"])
