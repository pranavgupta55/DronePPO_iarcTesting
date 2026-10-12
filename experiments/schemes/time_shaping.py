"""Time-optimal potential shaping (the project's best reward so far): -time_penalty + shaping_w * (Phi(next) - Phi(now)) with Phi = -time_to_go,
plus a spin penalty above a deadband."""
NAME = "time_shaping"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"time_penalty": 0.5, "shaping_w": 2000.0, "ang_vel_pen": 0.5, "deadband": 0.3}


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.phi = 0.0

    def reset(self, env):
        self.phi = -env.time_to_go()

    def on_target_spawn(self, env):
        self.phi = -env.time_to_go()  # re-baseline on the new target

    def set_progress(self, p):
        pass

    def extra_obs(self, env):
        return []

    def step(self, ctx):
        p = self.p
        phi = 0.0 if ctx.reached else -ctx.time_to_go  # a finished target has no time left
        r = -p["time_penalty"] + p["shaping_w"] * (phi - self.phi)
        self.phi = phi
        return r - p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
