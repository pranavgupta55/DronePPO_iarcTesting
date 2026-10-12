"""Torch-free pieces shared by training and evaluation: spec defaults, scheme loading, DroneEnv, numpy policy, env worker process.
Kept free of torch so the spawned worker processes start fast."""
import importlib
import math
import random
import traceback

import numpy as np

from core.config import (OBS_BASE_DIM, MAX_EXTRA_OBS, ACT_DIM, ENV_WIDTH, ENV_HEIGHT, ENV_DIAG, DRONE_SIZE, GRAVITY, TIME_STEP_DELTA, MAX_THRUSTER_FORCE,
                         THRUSTER_NEUTRAL_ANGLE, REACH_DIST, TIME_OPTIMAL_ACCEL, STRICT_REACH_SPEED, STRICT_REACH_ANG, STRICT_REACH_HOLD)
from core.drone import Drone

NORM_CLIP = 10.0
HOVER_THRUST = GRAVITY / MAX_THRUSTER_FORCE  # thruster level that cancels gravity when the drone is level
TOTAL_STEPS_PER_EPOCH = 32768  # default samples per epoch (4096 steps x 8 envs), split over however many envs there are

DEFAULTS = {
    "name": "run", "seed": 0, "scheme": "time_shaping", "params": {},
    "epochs": 300, "steps_per_epoch": None,  # per env; None = TOTAL_STEPS_PER_EPOCH / num_envs
    "episode_length": 4096, "num_envs": 8, "hidden": [16, 8],
    "pi_lr": 3e-4, "vf_lr": 1e-3, "lr_decay_end": 0.1, "lr_schedule": "linear",
    "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.2, "update_epochs": 4, "num_minibatches": 16, "init_log_std": -0.5,
    "reward_scaling": True, "obs_norm": True, "mirror": True, "torch_threads": 1,
    "reach_reward": 10000.0, "crash_penalty": 6000.0, "outward_w": 0.0, "outward_target": 0.5235987755982988,
    "far_target_min_dist": 400.0, "far_target_fraction_end": 0.7, "curriculum_end": 0.5, "target_margin": 100.0, "ramp_end": 0.5,
    "reach_speed": [50.0, 20.0], "reach_ang": [2.0, 0.5], "reach_hold": [1, 25],
}
NON_HYPER_KEYS = ("name", "seed", "scheme", "params")


def make_spec(spec):
    """Spec with defaults filled in; unknown keys are rejected so typos in a batch file fail loudly."""
    unknown = set(spec) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown spec keys: {sorted(unknown)}")
    full = {**DEFAULTS, **spec}
    full["params"] = dict(full["params"] or {})
    if full["steps_per_epoch"] is None:
        full["steps_per_epoch"] = math.ceil(TOTAL_STEPS_PER_EPOCH / full["num_envs"])
    return full


def load_scheme(name, params):
    """Imports schemes/<name>.py and builds its Scheme. Returns (module, scheme)."""
    module = importlib.import_module(f"schemes.{name}")
    n = int(module.EXTRA_OBS)
    if not 0 <= n <= MAX_EXTRA_OBS or len(module.EXTRA_OBS_MIRROR) != n:
        raise ValueError(f"scheme {name}: EXTRA_OBS must be 0..{MAX_EXTRA_OBS} with EXTRA_OBS_MIRROR of the same length")
    return module, module.Scheme(dict(params))


# Mirror symmetry (flipping left and right), as in ppo.py. Base layout: velX, velY, angVel, speed, relX, relY, T1Force, T2Force, T1Angle, T2Angle, sin, cos, velToTarget, hWall, vWall, hTTI, vTTI
BASE_MIRROR_PERM = [0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10, 11, 12, 13, 14, 15, 16]
BASE_MIRROR_SIGN = [-1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1]


def mirror_tools(extra_mirror):
    """(perm, sign) arrays that mirror an observation of OBS_BASE_DIM + len(extra_mirror) values."""
    n = len(extra_mirror)
    perm = np.array(BASE_MIRROR_PERM + list(range(OBS_BASE_DIM, OBS_BASE_DIM + n)))
    sign = np.array(BASE_MIRROR_SIGN + [float(s) for s in extra_mirror], dtype=np.float32)
    return perm, sign


def mirror_action(action):
    """Swaps the two thrusters; their angle changes also flip sign (left angle counts outward as +, the right as -)."""
    return np.array([action[1], action[0], -action[3], -action[2]])


class Ctx:
    """Per-step context handed to Scheme.step (plain floats, physical (unmirrored) frame). See schemes/README.md for the fields."""
    __slots__ = ("dist", "prev_dist", "dx", "dy", "vx", "vy", "speed", "approach_speed", "lateral_speed", "angle", "ang_vel", "prev_ang_vel",
                 "forces", "prev_forces", "angles_out", "action", "prev_action", "steps", "progress", "reached", "crashed", "hold_steps",
                 "time_to_go", "W", "H", "x", "y", "dist_to_wall")


class DroneEnv:
    def __init__(self, spec, module, scheme, max_steps=None):
        self.spec, self.scheme = spec, scheme
        self.max_steps = max_steps or spec["episode_length"]
        self.mirror_episodes = bool(spec["mirror"])
        self.mirrored = False
        self.n_extra = int(module.EXTRA_OBS)
        self.mirror_perm, self.mirror_sign = mirror_tools(module.EXTRA_OBS_MIRROR)
        self.obs_dim = OBS_BASE_DIM + self.n_extra
        self._on_spawn = getattr(scheme, "on_target_spawn", None)
        self.target_margin = spec["target_margin"]
        self.drone = Drone((ENV_WIDTH / 2, ENV_HEIGHT / 2), DRONE_SIZE, ENV_DIAG)
        self.progress = 1.0
        self.set_ramp(1.0)

    def set_ramp(self, progress):
        """Curricula (reach limits, far targets) as in ppo.py, plus the scheme's own progress. progress runs 0 to 1 over the session."""
        s = self.spec
        t = min(progress / s["ramp_end"], 1.0)
        lin = lambda ab: ab[0] + (ab[1] - ab[0]) * t
        self.progress = progress
        self.reach_speed, self.reach_ang_speed, self.reach_hold = lin(s["reach_speed"]), lin(s["reach_ang"]), round(lin(s["reach_hold"]))
        self.far_fraction = s["far_target_fraction_end"] * min(progress / s["curriculum_end"], 1.0)
        self.scheme.set_progress(progress)

    def set_eval_mode(self):
        """Strict reach rule, uniform targets, no mirroring (the fixed evaluation protocol)."""
        self.reach_speed, self.reach_ang_speed, self.reach_hold = STRICT_REACH_SPEED, STRICT_REACH_ANG, STRICT_REACH_HOLD
        self.far_fraction, self.mirror_episodes, self.target_margin = 0.0, False, 100.0

    def reset(self):
        d = self.drone
        d.reset()  # spawns at the centre with a small random tilt / velocity
        d.thruster_forces = [HOVER_THRUST, HOVER_THRUST]
        d.previous_thruster_forces = [HOVER_THRUST, HOVER_THRUST]
        self.mirrored = self.mirror_episodes and random.random() < 0.5
        self.steps, self.reaches, self.hold_steps = 0, 0, 0
        self.prev_action = np.zeros(ACT_DIM)
        self.prev_ang_vel = d.shell.angular_velocity
        self.spawn_target()
        self.prev_dist = self.dist_to_target()
        self.scheme.reset(self)
        return self.observe()

    def dist_to_target(self):
        c = self.drone.shell.center
        return math.hypot(self.target[0] - float(c[0]), self.target[1] - float(c[1]))

    def observe(self):
        extra = self.scheme.extra_obs(self) if self.n_extra else ()
        obs = self.drone.get_observation(self.target, extra)
        return obs[self.mirror_perm] * self.mirror_sign if self.mirrored else obs

    @staticmethod
    def _ttg(dist, speed):
        """Bang-bang time-optimal estimate (ppo.py): accelerate, then brake with the switch where remaining distance equals braking distance.
        speed is the velocity component toward the target. Overshooting is brake, then come back."""
        accel = TIME_OPTIMAL_ACCEL
        if dist >= speed * abs(speed) / (2 * accel):
            peak = math.sqrt(accel * dist + speed * speed / 2)
            return (2 * peak - speed) / accel
        return speed / accel + 2 * math.sqrt((speed * speed / (2 * accel) - dist) / accel)

    def time_to_go(self):
        shell = self.drone.shell
        to_x, to_y = self.target[0] - float(shell.center[0]), self.target[1] - float(shell.center[1])
        dist = math.hypot(to_x, to_y)
        vx, vy = (float(shell.vel[0, 0]) + float(shell.vel[1, 0])) / 2, (float(shell.vel[0, 1]) + float(shell.vel[1, 1])) / 2
        return self._ttg(dist, (vx * to_x + vy * to_y) / (dist + 1e-8))

    def spawn_target(self):
        """Uniform over the arena (with margin), except far_fraction of the time it must be at least far_target_min_dist from the drone
        (the farthest of 20 tries if none is that far)."""
        c = self.drone.shell.center
        drone_x, drone_y = float(c[0]), float(c[1])
        far = random.random() < self.far_fraction
        min_dist, m = self.spec["far_target_min_dist"], self.target_margin
        best, best_dist = None, -1.0
        for _ in range(20):
            cand = (random.uniform(m, ENV_WIDTH - m), random.uniform(m, ENV_HEIGHT - m))
            dist = math.hypot(cand[0] - drone_x, cand[1] - drone_y)
            if not far or dist >= min_dist:
                best = cand
                break
            if dist > best_dist:
                best, best_dist = cand, dist
        self.target = np.array(best)

    def step(self, action):
        """Returns (obs, reward, terminated, truncated). terminated = left the world, truncated = hit the step limit."""
        action = np.asarray(action, dtype=np.float64)
        if self.mirrored:
            action = mirror_action(action)  # everything below (and the scheme) lives in the physical frame
        d = self.drone
        d.apply_action(action, TIME_STEP_DELTA)
        d.physics_move(GRAVITY, TIME_STEP_DELTA)
        self.steps += 1

        shell = d.shell
        x, y = float(shell.center[0]), float(shell.center[1])
        dx, dy = float(self.target[0]) - x, float(self.target[1]) - y
        dist = math.hypot(dx, dy)
        vx, vy = (float(shell.vel[0, 0]) + float(shell.vel[1, 0])) / 2, (float(shell.vel[0, 1]) + float(shell.vel[1, 1])) / 2
        speed = math.hypot(vx, vy)
        approach = (vx * dx + vy * dy) / (dist + 1e-8)
        ang_vel = float(shell.angular_velocity)

        calm = dist < REACH_DIST and speed < self.reach_speed and abs(ang_vel) < self.reach_ang_speed
        self.hold_steps = self.hold_steps + 1 if calm else 0
        reached = self.hold_steps >= self.reach_hold
        crashed = not (0 <= x <= ENV_WIDTH and 0 <= y <= ENV_HEIGHT)

        c = Ctx()
        c.dist, c.prev_dist, c.dx, c.dy, c.vx, c.vy, c.speed = dist, self.prev_dist, dx, dy, vx, vy, speed
        c.approach_speed, c.lateral_speed = approach, math.sqrt(max(speed * speed - approach * approach, 0.0))
        c.angle, c.ang_vel, c.prev_ang_vel = float(shell.angle), ang_vel, self.prev_ang_vel
        c.forces, c.prev_forces = list(d.thruster_forces), list(d.previous_thruster_forces)
        c.angles_out = [d.thruster_angles[0] - THRUSTER_NEUTRAL_ANGLE, THRUSTER_NEUTRAL_ANGLE - d.thruster_angles[1]]  # > 0 = nozzle points outward
        c.action, c.prev_action = action, self.prev_action
        c.steps, c.progress, c.reached, c.crashed, c.hold_steps = self.steps, self.progress, reached, crashed, self.hold_steps
        c.time_to_go = self._ttg(dist, approach)
        c.W, c.H, c.x, c.y, c.dist_to_wall = ENV_WIDTH, ENV_HEIGHT, x, y, min(x, ENV_WIDTH - x, y, ENV_HEIGHT - y)

        reward = float(self.scheme.step(c))
        if self.spec["outward_w"]:  # overlay on any scheme: peaked reward for both nozzles pointing outward (same shape as ppo.py's)
            t = self.spec["outward_target"]
            reward += self.spec["outward_w"] * (max(-1.0, 1 - abs(c.angles_out[0] - t) / t) + max(-1.0, 1 - abs(c.angles_out[1] - t) / t)) / 2
        if not math.isfinite(reward):
            raise FloatingPointError(f"scheme {self.spec['scheme']} returned a non-finite reward ({reward}) at step {self.steps}")
        self.prev_action, self.prev_ang_vel = action, ang_vel

        if reached:
            self.hold_steps = 0
            reward += self.spec["reach_reward"]
            self.reaches += 1
            self.spawn_target()
            if self._on_spawn is not None:
                self._on_spawn(self)
            self.prev_dist = self.dist_to_target()
        else:
            self.prev_dist = dist
        if crashed:
            reward -= self.spec["crash_penalty"]
        return self.observe(), reward, crashed, self.steps >= self.max_steps


class NumpyPolicy:
    """The actor's forward pass in plain numpy, so a worker process can act without torch."""

    def __init__(self, layers, log_std):
        self.layers, self.std = layers, np.exp(log_std)  # layers is a list of (weight, bias) pairs
        self.log_norm = float(np.sum(log_std)) + 0.5 * len(log_std) * math.log(2 * math.pi)

    def mean(self, obs):
        x = obs
        for weight, bias in self.layers[:-1]:
            x = np.tanh(weight @ x + bias)
        return self.layers[-1][0] @ x + self.layers[-1][1]

    def act(self, obs, rng):
        """Samples an action and returns it with its log-probability."""
        mean = self.mean(obs)
        noise = rng.standard_normal(len(mean))
        return mean + self.std * noise, -0.5 * float(noise @ noise) - self.log_norm


def env_worker(conn, spec, rank):
    """Runs one DroneEnv in its own process. On request it acts with the policy it was sent for a whole stretch of steps, entirely on its own.
    Any exception is sent back as ("error", traceback) so the trainer fails with the real cause instead of hanging."""
    try:
        seed = spec["seed"] + rank
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        module, scheme = load_scheme(spec["scheme"], spec["params"])
        env = DroneEnv(spec, module, scheme)
        obs_dim = env.obs_dim
        obs, ep_return = env.reset(), 0.0
        while True:
            command, data = conn.recv()
            if command != "collect":
                break
            layers, log_std, obs_mean, obs_std, n_steps, progress = data
            env.set_ramp(progress)
            policy = NumpyPolicy(layers, log_std)
            raw_obs, acts = np.zeros((n_steps, obs_dim), dtype=np.float32), np.zeros((n_steps, ACT_DIM), dtype=np.float32)
            logp, rew = np.zeros(n_steps, dtype=np.float32), np.zeros(n_steps, dtype=np.float64)
            terminated, truncated = np.zeros(n_steps, dtype=bool), np.zeros(n_steps, dtype=bool)
            final_obs, episodes = [], []  # the last observation and the (return, length, reaches) of each episode that ends
            for t in range(n_steps):
                raw_obs[t] = obs
                acts[t], logp[t] = policy.act(np.clip((obs - obs_mean) / obs_std, -NORM_CLIP, NORM_CLIP), rng)
                obs, rew[t], terminated[t], truncated[t] = env.step(acts[t])
                ep_return += rew[t]
                if terminated[t] or truncated[t]:
                    final_obs.append(obs)
                    episodes.append((ep_return, env.steps, env.reaches))
                    ep_return = 0.0
                    obs = env.reset()
            conn.send((raw_obs, acts, logp, rew, terminated, truncated, np.array(final_obs, dtype=np.float32).reshape(-1, obs_dim), obs, episodes, (ep_return, env.steps)))
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
