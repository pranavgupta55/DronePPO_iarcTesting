"""Staged mixture: w_speed(p) * r_speed + safe_w * r_safe + w_nozzle(p) * nozzle_w * alignment. r_speed is the time-optimal shaping (time
penalty, potential, spin penalty); r_safe is a wall-proximity penalty; w_speed ramps up with progress p and w_nozzle ramps down to zero by nozzle_end."""
import math

from drone_rl.schemes.time_shaping import Scheme as TimeShaping
from drone_rl.schemes.thrust_weighted_nozzle_alignment import alignment

NAME = "stage_gated_reward_mixture"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

BASE = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3}
DEFAULTS = {**BASE, "speed_start": 0.3, "speed_ramp_end": 0.3, "safe_w": 1.0, "wall_ref": 60.0,
            "nozzle_w": 0.5, "nozzle_end": 0.7, "outward_target": math.pi / 6}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__({k: self.q[k] for k in BASE})
        self.progress = 0.0

    def set_progress(self, p):
        self.progress = p

    def step(self, ctx):
        q, p = self.q, self.progress
        w_speed = q["speed_start"] + (1.0 - q["speed_start"]) * min(1.0, p / max(q["speed_ramp_end"], 1e-6))
        w_nozzle = max(0.0, 1.0 - p / max(q["nozzle_end"], 1e-6))
        r_safe = -max(0.0, 1.0 - ctx.dist_to_wall / q["wall_ref"])
        r_nozzle = alignment(ctx.forces, ctx.angles_out, q["outward_target"])
        return w_speed * super().step(ctx) + q["safe_w"] * r_safe + w_nozzle * q["nozzle_w"] * r_nozzle
