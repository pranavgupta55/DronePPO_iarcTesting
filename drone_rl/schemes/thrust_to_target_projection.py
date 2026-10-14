"""Time-optimal shaping plus thrust-to-target projection: proj_w * sum_i F_i * cos(psi_i), where psi_i is the angle between thruster i's
push direction (opposite its exhaust) and the unit vector to the target. Rewards thrust that pushes the drone toward the target."""
import math

from drone_rl.config import THRUSTER_NEUTRAL_ANGLE
from drone_rl.schemes.time_shaping import Scheme as TimeShaping

NAME = "thrust_to_target_projection"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

BASE = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3}
DEFAULTS = {**BASE, "proj_w": 0.5}


def push_projection(forces, angles_out, body_angle, dx, dy):
    """sum_i F_i * cos(psi_i). The drone is pushed along -(cos a_i, sin a_i), with a_i the thruster's absolute angle (env/drone.py)."""
    theta = math.atan2(dy, dx)
    a1 = THRUSTER_NEUTRAL_ANGLE + angles_out[0] + body_angle  # angles_out[0] = thr0 - neutral
    a2 = THRUSTER_NEUTRAL_ANGLE - angles_out[1] + body_angle  # angles_out[1] = neutral - thr1
    return -forces[0] * math.cos(a1 - theta) - forces[1] * math.cos(a2 - theta)


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__({k: self.q[k] for k in BASE})

    def step(self, ctx):
        proj = push_projection(ctx.forces, ctx.angles_out, ctx.angle, ctx.dx, ctx.dy)
        return super().step(ctx) + self.q["proj_w"] * proj
