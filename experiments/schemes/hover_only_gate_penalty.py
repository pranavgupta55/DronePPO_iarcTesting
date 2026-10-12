"""time_shaping plus a tilt/spin penalty that only applies in the hover state: -hover_k * (w^2 + angle^2) * g_speed * g_dist, where
g_speed = sigmoid((v_low - speed) / v_soft) and g_dist = sigmoid((d_gate - dist) / d_soft). Both gates are smooth. Fast flight (speed above
v_low) and flight far from the target are not penalised for tilt. Near the target, while slow, tilt is penalised by design."""
import math

from schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "hover_only_gate_penalty"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "hover_k": 0.5, "v_low": 150.0, "v_soft": 30.0, "d_gate": 100.0, "d_soft": 20.0}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        gate = sigmoid((q["v_low"] - ctx.speed) / q["v_soft"]) * sigmoid((q["d_gate"] - ctx.dist) / q["d_soft"])
        return r - q["hover_k"] * (ctx.ang_vel * ctx.ang_vel + ctx.angle * ctx.angle) * gate
