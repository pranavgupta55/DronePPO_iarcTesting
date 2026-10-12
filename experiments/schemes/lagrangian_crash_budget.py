"""time_shaping plus a Lagrangian crash budget. The penalty weight lam multiplies a violation term (crash_w on a crash, wall_w while
closer than wall_margin px to a border). lam is a dual variable updated once per episode: lam += eta * (crashed_this_episode - budget),
clipped to [0, lam_max]. The env's own crash penalty still applies on top. lam lives per Scheme instance (one per env worker)."""
from schemes.time_shaping import Scheme as TimeShaping, DEFAULTS as TS_DEFAULTS

NAME = "lagrangian_crash_budget"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {**TS_DEFAULTS, "shaping_w": 100.0, "lam0": 1.0, "eta": 0.01, "budget": 0.02, "lam_max": 100.0,
            "crash_w": 10.0, "wall_w": 0.05, "wall_margin": 20.0}


class Scheme(TimeShaping):
    def __init__(self, params):
        self.q = {**DEFAULTS, **params}
        super().__init__(self.q)
        self.lam = float(self.q["lam0"])
        self.crashed_ep = False
        self.episode_open = False  # the dual update needs a finished episode, so skip it on the very first reset

    def reset(self, env):
        q = self.q
        if self.episode_open:
            crash = 1.0 if self.crashed_ep else 0.0
            self.lam = min(q["lam_max"], max(0.0, self.lam + q["eta"] * (crash - q["budget"])))
        self.episode_open = True
        self.crashed_ep = False
        super().reset(env)

    def step(self, ctx):
        q = self.q
        r = super().step(ctx)
        if ctx.crashed:
            self.crashed_ep = True
            viol = q["crash_w"]
        elif ctx.dist_to_wall < q["wall_margin"]:
            viol = q["wall_w"]
        else:
            viol = 0.0
        return r - self.lam * viol
