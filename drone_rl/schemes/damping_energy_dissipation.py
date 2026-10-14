"""time_shaping plus a rotational-energy penalty: -damp_k * max(0, w^2 - w_prev^2) while speed < v_low (hard gate). Only increases of
spin energy are charged, so the drone may damp spin for free. Tilt itself is not penalised."""
from drone_rl.schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "damping_energy_dissipation"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "damp_k": 5.0, "v_low": 150.0}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        gain = max(0.0, ctx.ang_vel * ctx.ang_vel - ctx.prev_ang_vel * ctx.prev_ang_vel)
        gate = 1.0 if ctx.speed < q["v_low"] else 0.0
        return r - q["damp_k"] * gain * gate
