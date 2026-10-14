"""Time-optimal shaping plus the thrust-weighted nozzle alignment (idea 22), with its weight falling linearly from nozzle_w at progress 0
to zero at progress zero_at. Nozzle discipline is taught early and dropped late, so speed is free at the end."""
import math

from drone_rl.schemes.time_shaping import Scheme as TimeShaping
from drone_rl.schemes.thrust_weighted_nozzle_alignment import alignment

NAME = "nozzle_annealed_curriculum"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

BASE = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3}
DEFAULTS = {**BASE, "nozzle_w": 0.5, "zero_at": 0.7, "outward_target": math.pi / 6}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__({k: self.q[k] for k in BASE})
        self.progress = 0.0

    def set_progress(self, p):
        self.progress = p

    def step(self, ctx):
        q = self.q
        w = q["nozzle_w"] * max(0.0, 1.0 - self.progress / max(q["zero_at"], 1e-6))
        return super().step(ctx) + w * alignment(ctx.forces, ctx.angles_out, q["outward_target"])
