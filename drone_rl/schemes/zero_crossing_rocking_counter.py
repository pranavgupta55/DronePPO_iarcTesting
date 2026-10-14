"""time_shaping plus a rocking penalty: count spin sign flips (ignoring |w| < w_eps) over a rolling window of `window` steps, then pay
-flip_k * flips per step while speed < v_low (hard gate). Each flip therefore costs about flip_k * window in total."""
from collections import deque

from drone_rl.schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "zero_crossing_rocking_counter"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "flip_k": 0.05, "window": 60, "w_eps": 0.05, "v_low": 150.0}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)
        self._reset_window()

    def _reset_window(self):
        self.hist = deque(maxlen=int(self.q["window"]))
        self.count = 0
        self.last_sign = 0

    def reset(self, env):
        super().reset(env)
        self._reset_window()

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        w = ctx.ang_vel
        sign = 0 if abs(w) < q["w_eps"] else (1 if w > 0 else -1)
        flip = 1 if (sign != 0 and self.last_sign != 0 and sign != self.last_sign) else 0
        if sign != 0:
            self.last_sign = sign
        if len(self.hist) == self.hist.maxlen:
            self.count -= self.hist[0]  # the flip about to fall out of the window
        self.hist.append(flip)
        self.count += flip
        gate = 1.0 if ctx.speed < q["v_low"] else 0.0
        return r - q["flip_k"] * self.count * gate
