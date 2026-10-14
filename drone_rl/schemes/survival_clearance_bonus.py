"""time_shaping plus a small per-step clearance bonus: clear_c * (1 - anneal * progress) * min(1, dist_to_wall / (clear_frac * min(W, H))).
Tilt is not penalised. clear_c stays below time_penalty so that staying alive never pays more than flying to the target."""
from drone_rl.schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "survival_clearance_bonus"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "clear_c": 0.1, "clear_frac": 0.25, "anneal": 0.5}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        clearance = min(1.0, ctx.dist_to_wall / (q["clear_frac"] * min(ctx.W, ctx.H)))
        return r + q["clear_c"] * (1.0 - q["anneal"] * ctx.progress) * clearance
