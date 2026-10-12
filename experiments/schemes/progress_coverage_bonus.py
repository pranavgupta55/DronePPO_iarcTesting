"""Time-optimal shaping plus a coverage bonus: each entry into a grid cell (8x6 over the world) while moving faster than v_thresh pays
cov_w / sqrt(visits + 1), where visits counts earlier entries this episode. Re-entries past the first also pay -revisit_pen * cov_w."""
import math

from core.config import ENV_WIDTH, ENV_HEIGHT
from schemes.time_shaping import Scheme as TimeShaping

NAME = "progress_coverage_bonus"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

NX, NY = 8, 6
BASE = {"time_penalty": 0.5, "shaping_w": 50.0, "ang_vel_pen": 0.5, "deadband": 0.3}
DEFAULTS = {**BASE, "cov_w": 1.0, "v_thresh": 60.0, "revisit_pen": 0.5}


def cell_index(x, y):
    gx = min(NX - 1, max(0, int(x / ENV_WIDTH * NX)))
    gy = min(NY - 1, max(0, int(y / ENV_HEIGHT * NY)))
    return gy * NX + gx


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__({k: self.q[k] for k in BASE})
        self.visits = [0] * (NX * NY)
        self.cell = 0

    def reset(self, env):
        super().reset(env)
        c = env.drone.shell.center
        self.visits = [0] * (NX * NY)
        self.cell = cell_index(float(c[0]), float(c[1]))
        self.visits[self.cell] = 1  # the starting cell counts as visited

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        cell = cell_index(ctx.x, ctx.y)
        if cell != self.cell:
            v = self.visits[cell]
            if ctx.speed > q["v_thresh"]:
                r += q["cov_w"] / math.sqrt(v + 1) - (q["revisit_pen"] * q["cov_w"] if v >= 1 else 0.0)
            self.visits[cell] = v + 1
            self.cell = cell
        return r
