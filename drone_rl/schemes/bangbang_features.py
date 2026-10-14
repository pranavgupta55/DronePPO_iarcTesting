"""Reward of time_to_wall_barrier (world size set to the current world) plus 7 extra observations that expose the quantities the hand-written bang-bang controller
decides with, so a [24,16,8,4] network can imitate it: speed along / across the line to the target, the braking margin (distance left minus stopping distance),
and the body-angle error to the thrust direction needed for accelerating and for braking (sin, cos)."""
import math

from drone_rl.controller import DEFAULTS as BB
from drone_rl.config import GRAVITY
from drone_rl.schemes import time_to_wall_barrier as base

NAME = "bangbang_features"
EXTRA_OBS = 7
EXTRA_OBS_MIRROR = [1, -1, 1, -1, 1, -1, 1]  # the across-speed and the sines flip under a left-right mirror


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Scheme(base.Scheme):
    def extra_obs(self, env):
        sh = env.drone.shell
        to_x, to_y = env.target[0] - float(sh.center[0]), env.target[1] - float(sh.center[1])
        dist = math.hypot(to_x, to_y) + 1e-9
        ux, uy = to_x / dist, to_y / dist
        vx, vy = (float(sh.vel[0, 0]) + float(sh.vel[1, 0])) / 2, (float(sh.vel[0, 1]) + float(sh.vel[1, 1])) / 2
        along, across = vx * ux + vy * uy, ux * vy - uy * vx
        braking_distance = BB["margin"] * along * abs(along) / (2 * BB["a_brake"])
        err_accel = _wrap(math.atan2(BB["a_acc"] * ux, -(BB["a_acc"] * uy - GRAVITY)) - sh.angle)  # thrust direction for accelerating toward the target, relative to the body
        err_brake = _wrap(math.atan2(-BB["a_brake"] * ux, -(-BB["a_brake"] * uy - GRAVITY)) - sh.angle)
        return [math.tanh(along / 400.0), math.tanh(across / 400.0), math.tanh((dist - braking_distance) / 300.0),
                math.sin(err_accel), math.cos(err_accel), math.sin(err_brake), math.cos(err_brake)]
