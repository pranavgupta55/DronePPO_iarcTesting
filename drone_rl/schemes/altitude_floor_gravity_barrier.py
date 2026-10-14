"""time_shaping plus a floor-only gravity barrier: -barrier_k * vdown^2 * exp(-h / h0), with h = distance above the floor and vdown = max(0, vy).
In this env gravity pulls toward +y, so the floor is at y = H and downward speed is vy > 0. Only falling speed near the floor is penalised;
side walls and tilt are not."""
import math

from drone_rl.schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "altitude_floor_gravity_barrier"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "barrier_k": 1e-5, "h0": 100.0}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        h = max(ctx.H - ctx.y, 0.0)  # height above the floor (floor at y = H)
        vdown = max(ctx.vy, 0.0)  # falling speed (positive vy points at the floor)
        return r - q["barrier_k"] * vdown * vdown * math.exp(-h / q["h0"])
