"""Idea 3, Lyapunov-style time potential. V = time_to_go + c_va * approach_speed (V = 0 on a finished target). Reward = -w * dV/dt, with the
rate clipped to +-rate_clip for robustness at direction reversals. Plus time penalty and spin penalty above a deadband."""
import math

NAME = "lyapunov_time_potential"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"w": 1.0, "c_va": 0.001, "rate_clip": 5.0, "dt": 0.02, "time_penalty": 0.1, "ang_vel_pen": 0.5, "deadband": 0.3}


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self.v = 0.0

    def _V(self, env, tg, va):
        return tg + self.p["c_va"] * va

    def reset(self, env):
        self.v = self._V(env, env.time_to_go(), _approach(env))

    def on_target_spawn(self, env):
        self.reset(env)

    def set_progress(self, p):
        pass

    def extra_obs(self, env):
        return []

    def step(self, ctx):
        p = self.p
        v = 0.0 if ctx.reached else ctx.time_to_go + p["c_va"] * ctx.approach_speed  # a finished target has no time left
        rate = max(-p["rate_clip"], min(p["rate_clip"], (v - self.v) / p["dt"]))
        self.v = v
        r = -p["w"] * rate - p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        return r


def _approach(env):
    shell = env.drone.shell
    cx, cy = float(shell.center[0]), float(shell.center[1])
    vx = (float(shell.vel[0, 0]) + float(shell.vel[1, 0])) / 2
    vy = (float(shell.vel[0, 1]) + float(shell.vel[1, 1])) / 2
    dx, dy = float(env.target[0]) - cx, float(env.target[1]) - cy
    return (vx * dx + vy * dy) / (math.hypot(dx, dy) + 1e-8)
