"""Idea 5, entry speed then stop. Once per target, on first entry inside `radius`: entry_k * clip(approach / vel_ref, 0, 1). Inside the radius,
stop_k * clip(speed / vel_ref, 0, 1) per step (stop term). Leaving the radius before hold_need consecutive steps inside costs leave_k. Dense part:
approach speed, time penalty, spin penalty."""

NAME = "entry_speed_fly_through"
EXTRA_OBS = 0
EXTRA_OBS_MIRROR = []

DEFAULTS = {"radius": 20.0, "entry_k": 5.0, "stop_k": 0.5, "leave_k": 5.0, "hold_need": 25, "vel_ref": 150.0, "approach_w": 0.5,
            "time_penalty": 0.1, "ang_vel_pen": 0.5, "deadband": 0.3}


class Scheme:
    def __init__(self, params):
        self.p = {**DEFAULTS, **params}
        self._new_target()

    def _new_target(self):
        self.entered = False
        self.inside = False
        self.inside_steps = 0

    def reset(self, env):
        self._new_target()

    def on_target_spawn(self, env):
        self._new_target()

    def set_progress(self, p):
        pass

    def extra_obs(self, env):
        return []

    def step(self, ctx):
        p = self.p
        vel_ref = p["vel_ref"]
        r = p["approach_w"] * max(-1.0, min(1.0, ctx.approach_speed / vel_ref)) - p["time_penalty"]
        r -= p["ang_vel_pen"] * max(0.0, abs(ctx.ang_vel) - p["deadband"])
        if ctx.dist < p["radius"]:
            if not self.entered:
                self.entered = True
                r += p["entry_k"] * max(0.0, min(1.0, ctx.approach_speed / vel_ref))
            r -= p["stop_k"] * min(1.0, ctx.speed / vel_ref)
            self.inside = True
            self.inside_steps += 1
        elif self.inside:
            if self.inside_steps < p["hold_need"] and not ctx.reached:
                r -= p["leave_k"]  # left before the hold window was completed
            self.inside = False
            self.inside_steps = 0
        return r
