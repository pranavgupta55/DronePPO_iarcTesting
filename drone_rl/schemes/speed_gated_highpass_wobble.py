"""time_shaping plus a high-pass wobble penalty gated by low speed: lp = (1 - a) * lp + a * w (EMA of spin), hp = w - lp,
penalty -wobble_k * hp^2 * max(0, 1 - speed / v_low). Fast flight, where spin is intentional, is not affected."""
from drone_rl.schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "speed_gated_highpass_wobble"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "wobble_k": 1.0, "lp_alpha": 0.1, "v_low": 150.0}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)
        self.lp = 0.0

    def reset(self, env):
        super().reset(env)
        self.lp = float(env.drone.shell.angular_velocity)

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        w = ctx.ang_vel
        self.lp = (1.0 - q["lp_alpha"]) * self.lp + q["lp_alpha"] * w
        hp = w - self.lp
        gate = max(0.0, 1.0 - ctx.speed / q["v_low"])
        return r - q["wobble_k"] * hp * hp * gate
