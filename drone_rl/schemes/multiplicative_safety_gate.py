"""Time-optimal shaping where the progress (potential) term is multiplied by a wall-clearance gate sigmoid((wall - wall_mid) / wall_scale).
The time penalty and spin penalty are not gated, so the gate cannot be used to dodge them. Near walls the speed shaping switches off."""
import math

NAME = "multiplicative_safety_gate"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3, "wall_mid": 40.0, "wall_scale": 10.0}


def sigmoid(x):
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


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
        gate = sigmoid((ctx.dist_to_wall - p["wall_mid"]) / p["wall_scale"])
        shaping = p["shaping_w"] * gate * (phi - self.phi)
        self.phi = phi
        return shaping - p["time_penalty"] - p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
