"""time_shaping plus an action smoothness penalty gated by low speed: -chatter_k * |a - a_prev|^2 * max(0, 1 - speed / v_low), with the
4-dim action in the physical frame. Bang-bang thrust at speed is not charged; chatter while nearly stopped is."""
from schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "action_chatter_at_low_speed"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "chatter_k": 0.02, "v_low": 150.0}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        da = 0.0
        for a, b in zip(ctx.action, ctx.prev_action):
            d = float(a) - float(b)
            da += d * d
        gate = max(0.0, 1.0 - ctx.speed / q["v_low"])
        return r - q["chatter_k"] * da * gate
