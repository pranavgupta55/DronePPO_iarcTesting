"""time_shaping plus a crash cost that scales with the time the crash throws away: on a crash, extra -crash_k * clip(time_to_go, t_floor, t_cap)
(seconds). The env's flat crash penalty still applies on top. Late crashes (little time-to-go left) cost less than early ones."""
from drone_rl.schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "crash_cost_scaled_by_remaining_value"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "crash_k": 1.0, "t_floor": 1.0, "t_cap": 10.0}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)

    def step(self, ctx):
        r = super().step(ctx)
        if ctx.crashed:
            q = self.q
            r -= q["crash_k"] * min(max(ctx.time_to_go, q["t_floor"]), q["t_cap"])
        return r
