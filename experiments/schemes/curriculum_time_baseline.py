"""Idea 7, curriculum time baseline. Targets are bucketed by spawn distance (bucket_px wide, n_buckets buckets). Each bucket keeps a fast EMA
of reach time B. On a reach the reward is base_k * clip((B - t) / B, -1, 1) (zero-mean advantage-like signal, fair across distances), then B moves
toward t by ema. Dense part: approach speed, time penalty, spin penalty."""

NAME = "curriculum_time_baseline"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"base_k": 10.0, "ema": 0.2, "bucket_px": 150.0, "n_buckets": 10, "min_base_s": 0.05, "dt": 0.02, "approach_w": 1.0, "vel_ref": 150.0,
            "time_penalty": 0.1, "ang_vel_pen": 0.5, "deadband": 0.3}


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.baseline = {}  # bucket -> EMA of reach time (s), kept across episodes
        self.bucket = 0
        self.steps_since = 0

    def _spawn(self, env):
        self.steps_since = 0
        d0 = env.dist_to_target()
        self.bucket = min(int(d0 // self.p["bucket_px"]), self.p["n_buckets"] - 1)

    def reset(self, env):
        self._spawn(env)

    def on_target_spawn(self, env):
        self._spawn(env)

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
            base = self.baseline.get(self.bucket)
            if base is None:
                self.baseline[self.bucket] = t
            else:
                rel = (base - t) / max(base, p["min_base_s"])
                r += p["base_k"] * max(-1.0, min(1.0, rel))
                self.baseline[self.bucket] = base + p["ema"] * (t - base)
        return r
