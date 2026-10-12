"""Idea 6, self-record split bonus. Keeps a best split time (seconds since spawn) per target index within the episode, across episodes. On a reach
it pays record_k * max(0, best - t) when the split beats the record, then moves the record to min(best, t), or drifts it up by ema when slower
(so an old lucky record decays). Records are floored at floor_s. Dense part: approach speed, time penalty, spin penalty."""

NAME = "self_record_split_bonus"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"record_k": 10.0, "ema": 0.05, "floor_s": 0.2, "max_keys": 10, "dt": 0.02, "approach_w": 1.0, "vel_ref": 150.0,
            "time_penalty": 0.1, "ang_vel_pen": 0.5, "deadband": 0.3}


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.best = {}  # target index -> best split seconds, kept across episodes
        self.k = 0
        self.steps_since = 0

    def reset(self, env):
        self.k = 0
        self.steps_since = 0

    def on_target_spawn(self, env):
        self.steps_since = 0

    def set_progress(self, p):
        pass

    def extra_obs(self, env):
        return []

    def step(self, ctx):
        p = self.p
        self.steps_since += 1
        r = p["approach_w"] * max(-1.0, min(1.0, ctx.approach_speed / p["vel_ref"])) - p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        if ctx.reached:
            t = self.steps_since * p["dt"]
            key = min(self.k, p["max_keys"] - 1)
            best = self.best.get(key)
            if best is None:
                self.best[key] = max(t, p["floor_s"])  # first record: nothing to beat yet
            else:
                r += p["record_k"] * max(0.0, best - t)
                new = t if t < best else best + p["ema"] * (t - best)
                self.best[key] = max(new, p["floor_s"])
            self.k += 1
        return r
