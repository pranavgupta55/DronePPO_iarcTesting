"""Time-optimal shaping plus a potential-based nozzle term: Phi = -(|out1 - target| + |out2 - target|) in radians,
reward nozzle_w * (gamma * Phi(next) - Phi(now)). Pays for turning the nozzles toward the outward target, whether or not thrust is on."""
import math

from drone_rl.config import THRUSTER_NEUTRAL_ANGLE
from drone_rl.schemes.time_shaping import Scheme as TimeShaping

NAME = "nozzle_alignment_potential"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

BASE = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3}
DEFAULTS = {**BASE, "nozzle_w": 2.0, "gamma": 0.99, "outward_target": math.pi / 6}


def nozzle_potential(angles_out, target):
    """Phi <= 0, maximal (0) when both nozzles sit at the outward target."""
    return -sum(abs(a - target) for a in angles_out)


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__({k: self.q[k] for k in BASE})
        self.nphi = 0.0

    def reset(self, env):
        super().reset(env)
        t = env.drone.thruster_angles
        out = [t[0] - THRUSTER_NEUTRAL_ANGLE, THRUSTER_NEUTRAL_ANGLE - t[1]]  # same convention as ctx.angles_out
        self.nphi = nozzle_potential(out, self.q["outward_target"])

    def step(self, ctx):
        q = self.q
        nphi = nozzle_potential(ctx.angles_out, q["outward_target"])
        r = q["nozzle_w"] * (q["gamma"] * nphi - self.nphi)
        self.nphi = nphi
        return super().step(ctx) + r
