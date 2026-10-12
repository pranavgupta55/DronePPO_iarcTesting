"""time_shaping plus a settle bonus inside the reach zone: while dist < radius and the hold counter is running, pay
settle_k * min(1, hold / hold_max) * exp(-|w| / w0). It grows as the drone holds still, so settling pays more than idling. The
hold counter itself only runs on calm steps (env rule), so this is zero unless the drone is nearly stopped at the target."""
import math

from schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "settle_bonus_in_hold"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "settle_k": 1.0, "hold_max": 25.0, "w0": 0.5, "radius": 20.0}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        if ctx.dist < q["radius"] and ctx.hold_steps > 0:
            frac = min(1.0, ctx.hold_steps / q["hold_max"])
            r += q["settle_k"] * frac * math.exp(-abs(ctx.ang_vel) / q["w0"])
        return r
