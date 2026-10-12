"""Time-optimal shaping plus a penalty on thrust imbalance |F1 - F2| while hovering (slow and close to the target). Fast flight is not affected."""
from schemes.time_shaping import Scheme as TimeShaping

NAME = "thrust_asymmetry_at_hover"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

BASE = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3}
# hover gate: linear ramps to 0 at speed v_low (px/s) and at distance d_low (px)
DEFAULTS = {**BASE, "asym_w": 0.5, "v_low": 30.0, "d_low": 100.0}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__({k: self.q[k] for k in BASE})

    def step(self, ctx):
        q = self.q
        gate = max(0.0, 1.0 - ctx.speed / q["v_low"]) * max(0.0, 1.0 - ctx.dist / q["d_low"])
        return super().step(ctx) - q["asym_w"] * abs(ctx.forces[0] - ctx.forces[1]) * gate
