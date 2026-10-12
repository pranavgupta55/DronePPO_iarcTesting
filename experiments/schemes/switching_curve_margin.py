"""Idea 1, switching-curve braking margin. sigma = d - va^2/(2*amax) (va = closing speed, 0 if moving away). Penalise sigma < 0 (the drone
cannot stop before the target at full braking), plus an approach-speed term so it does not creep, a time penalty and a spin penalty above a deadband.
Extra obs: clip(sigma / sigma_ref, -1, 1)."""
import math

NAME = "switching_curve_margin"
EXTRA_OBS = 1
EXTRA_OBS_MIRROR = [1]  # distance-based, unchanged by a left/right mirror

DEFAULTS = {"amax": 6000.0, "viol_w": 0.01, "approach_w": 1.0, "vel_ref": 150.0, "sigma_ref": 300.0, "time_penalty": 0.1, "ang_vel_pen": 0.5, "deadband": 0.3}


def _margin(d, va, amax):
    """Braking margin; only a closing drone (va > 0) has a non-zero braking distance, so moving away gives sigma = d > 0 (no penalty)."""
    return d - max(va, 0.0) ** 2 / (2 * amax)


def _closing(env):
    shell = env.drone.shell
    cx, cy = float(shell.center[0]), float(shell.center[1])
    vx = (float(shell.vel[0, 0]) + float(shell.vel[1, 0])) / 2
    vy = (float(shell.vel[0, 1]) + float(shell.vel[1, 1])) / 2
    dx, dy = float(env.target[0]) - cx, float(env.target[1]) - cy
    d = math.hypot(dx, dy)
    return d, (vx * dx + vy * dy) / (d + 1e-8)


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}

    def reset(self, env):
        pass

    def set_progress(self, p):
        pass

    def extra_obs(self, env):
        d, va = _closing(env)
        return [max(-1.0, min(1.0, _margin(d, va, self.p["amax"]) / self.p["sigma_ref"]))]

    def step(self, ctx):
        p = self.p
        sigma = _margin(ctx.dist, ctx.approach_speed, p["amax"])
        r = -p["viol_w"] * max(0.0, -sigma)
        r += p["approach_w"] * max(-1.0, min(1.0, ctx.approach_speed / p["vel_ref"]))
        r -= p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        return r
