"""Idea 2, time saved per target. On each reach pay save_k * (T_ref - t_elapsed) seconds, with T_ref = time_to_go at spawn * factor and factor
annealed from factor_start (loose) to factor_end (tight) with training progress. Dense part: approach speed, time penalty, spin penalty."""
import math

NAME = "time_saved_per_target"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"save_k": 10.0, "factor_start": 3.0, "factor_end": 1.1, "min_ref": 0.2, "dt": 0.02, "approach_w": 1.0, "vel_ref": 150.0,
            "time_penalty": 0.1, "ang_vel_pen": 0.5, "deadband": 0.3}


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.progress = 0.0
        self.tg0 = 1.0
        self.steps_since = 0

    def reset(self, env):
        self.steps_since = 0
        self.tg0 = env.time_to_go()

    def on_target_spawn(self, env):
        self.reset(env)

    def set_progress(self, p):
        self.progress = p

    def extra_obs(self, env):
        return []

    def step(self, ctx):
        p = self.p
        self.steps_since += 1
        r = p["approach_w"] * max(-1.0, min(1.0, ctx.approach_speed / p["vel_ref"])) - p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        if ctx.reached:
            factor = p["factor_start"] + (p["factor_end"] - p["factor_start"]) * self.progress
            t_ref = max(self.tg0, p["min_ref"]) * factor
            r += p["save_k"] * (t_ref - self.steps_since * p["dt"])
        return r
