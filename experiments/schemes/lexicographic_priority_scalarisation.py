"""Lexicographic priorities: r_safe is a wall-proximity penalty. If r_safe > -safe_eps the step reward is r_speed (time-optimal shaping, time
penalty, spin penalty); otherwise it is r_safe alone, so near a wall only safety counts. No weights are traded between the two."""
import math

NAME = "lexicographic_priority_scalarisation"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3, "wall_ref": 60.0, "wall_w": 1.0, "safe_eps": 0.05}


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.phi = 0.0

    def reset(self, env):
        self.phi = -env.time_to_go()

    def on_target_spawn(self, env):
        self.phi = -env.time_to_go()

    def set_progress(self, p):
        pass

    def extra_obs(self, env):
        return []

    def step(self, ctx):
        p = self.p
        phi = 0.0 if ctx.reached else -ctx.time_to_go
        r_speed = p["shaping_w"] * (phi - self.phi) - p["time_penalty"] - p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        self.phi = phi  # state is kept current even when the safety branch wins
        r_safe = -p["wall_w"] * max(0.0, 1.0 - ctx.dist_to_wall / p["wall_ref"])
        return r_speed if r_safe > -p["safe_eps"] else r_safe
