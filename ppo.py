import argparse
import csv
import json
import math
import multiprocessing as mp
import random
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from drone import Drone
from config import (OBS_DIM, ACT_DIM, ENV_WIDTH, ENV_HEIGHT, ENV_DIAG, DRONE_SIZE, GRAVITY, TIME_STEP_DELTA, MAX_THRUSTER_FORCE,
                    THRUSTER_NEUTRAL_ANGLE, TARGET_MARGIN, REACH_DIST, REACH_SPEED_START, REACH_SPEED_END, REACH_ANGULAR_SPEED_START, REACH_ANGULAR_SPEED_END, REACH_HOLD_STEPS_START, REACH_HOLD_STEPS_END,
                    TIME_SHAPING_REWARD, TIME_PENALTY, TIME_OPTIMAL_ACCEL, OUTWARD_REWARD, OUTWARD_TARGET, REACH_REWARD, CRASH_PENALTY,
                    SMOOTHNESS_RAMP_START, SMOOTHNESS_RAMP_END, ACTION_CHANGE_PENALTY, ANGULAR_ACCEL_PENALTY,
                    REWARD_RAMP_END, FAR_TARGET_MIN_DIST, FAR_TARGET_FRACTION_END, TARGET_CURRICULUM_END, SLOW_TILT_EXTRA, SLOW_TILT_SPEED, LEVEL_ANGLE, TILT_PENALTY_MIN, TILT_PENALTY_MAX_START, TILT_PENALTY_MAX_END, TILT_PENALTY_RAMP_STEPS, TILT_PENALTY_DECAY,
                    LEVEL_REWARD_MAX_START, LEVEL_REWARD_MAX_END, LEVEL_REWARD_RAMP_STEPS, LEVEL_REWARD_DECAY, ANGULAR_VELOCITY_PENALTY, ANGULAR_VELOCITY_DEADBAND)

EPOCHS = 250
STEPS_PER_EPOCH = 4096  # steps each environment takes per epoch (the epoch's total is this times NUM_ENVS)
SAVE_EVERY = 20  # epochs between checkpoints
SNAPSHOT_EVERY = 500  # epochs between kept snapshots of the model (the checkpoint files are overwritten, the snapshots are not)
MAX_EPISODE_STEPS = 4096
HOVER_THRUST = GRAVITY / MAX_THRUSTER_FORCE  # thruster level that cancels gravity when the drone is level
HIDDEN_SIZES = (16, 8)
GAMMA, GAE_LAMBDA, CLIP_RATIO = 0.99, 0.95, 0.2
PI_LR, VF_LR = 3e-4, 1e-3  # the critic learns faster than the actor
LR_DECAY_END = 0.1  # learning rates fall to this fraction of their start value by the last epoch
LR_SCHEDULE = "linear"  # "linear" or "exponential" (exponential drops fast early: to 0.1x by the halfway point when the end value is 0.01)
UPDATE_EPOCHS, NUM_MINIBATCHES = 4, 16  # passes over each epoch's batch, and how many minibatches the batch is split into
NUM_ENVS = 8  # environments stepped together, each in its own process
NORM_CLIP = 10.0  # normalized observations and rewards are clipped to +/- this


def mlp(out_dim, hidden_sizes):
    sizes = [OBS_DIM, *hidden_sizes]
    layers = []
    for n_in, n_out in zip(sizes, sizes[1:]):
        layers += [nn.Linear(n_in, n_out), nn.Tanh()]
    return nn.Sequential(*layers, nn.Linear(sizes[-1], out_dim))


class Actor(nn.Module):
    def __init__(self, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.net = mlp(ACT_DIM, hidden_sizes)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

    def dist(self, obs):
        return Normal(self.net(obs), self.log_std.exp())


class Critic(nn.Module):
    def __init__(self, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.net = mlp(1, hidden_sizes)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# Mirror symmetry (flipping left and right). The simulation, observations and rewards are exactly mirror symmetric (checked step by step), so
# showing the policy a randomly mirrored version of each episode makes it learn left and right equally.
# Observation layout: velX, velY, angVel, speed, relX, relY, T1Force, T2Force, T1Angle, T2Angle, sin, cos, velToTarget, hWall, vWall, hTTI, vTTI, tiltWeight
OBS_MIRROR_PERM = np.array([0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17])
OBS_MIRROR_SIGN = np.array([-1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)


def mirror_obs(obs):
    return obs[OBS_MIRROR_PERM] * OBS_MIRROR_SIGN


def mirror_action(action):
    """Swaps the two thrusters; their angle changes also flip sign, because the left angle counts outward as + and the right as -."""
    return np.array([action[1], action[0], -action[3], -action[2]])


class DroneEnv:
    def __init__(self, max_steps=MAX_EPISODE_STEPS, mirror_episodes=True):
        self.max_steps = max_steps
        self.mirror_episodes = mirror_episodes
        self.mirrored = False
        self.set_ramp(1.0)  # the end of every curriculum unless a trainer says otherwise
        self.drone = Drone((ENV_WIDTH / 2, ENV_HEIGHT / 2), DRONE_SIZE, ENV_DIAG)

    def reset(self):
        self.drone.reset()  # spawns at the center with a small random tilt / velocity
        self.drone.thruster_forces = [HOVER_THRUST, HOVER_THRUST]
        self.mirrored = self.mirror_episodes and random.random() < 0.5
        self.steps = 0
        self.reaches = 0
        self.prev_action = None
        self.hold_steps = 0
        self.prev_ang_vel = self.drone.shell.angular_velocity
        self.tilt_weight = TILT_PENALTY_MIN
        self.level_weight = 0.0
        self.spawn_target()
        self.potential = -self.time_to_go()
        return self.observe()

    def set_ramp(self, progress):
        """Moves the level reward and tilt penalty caps from their strict start values to their relaxed end values as the session progresses (0 to 1)."""
        t = min(progress / REWARD_RAMP_END, 1.0)
        self.level_max = LEVEL_REWARD_MAX_START + (LEVEL_REWARD_MAX_END - LEVEL_REWARD_MAX_START) * t
        self.tilt_max = TILT_PENALTY_MAX_START + (TILT_PENALTY_MAX_END - TILT_PENALTY_MAX_START) * t
        self.reach_speed = REACH_SPEED_START + (REACH_SPEED_END - REACH_SPEED_START) * t  # how still and unspinning the drone must be to count as arrived
        self.reach_ang_speed = REACH_ANGULAR_SPEED_START + (REACH_ANGULAR_SPEED_END - REACH_ANGULAR_SPEED_START) * t
        self.reach_hold = round(REACH_HOLD_STEPS_START + (REACH_HOLD_STEPS_END - REACH_HOLD_STEPS_START) * t)  # steps in a row it must stay calm at the target
        self.far_fraction = FAR_TARGET_FRACTION_END * min(progress / TARGET_CURRICULUM_END, 1.0)  # share of new targets that must be far away
        self.smooth_scale = min(max((progress - SMOOTHNESS_RAMP_START) / (SMOOTHNESS_RAMP_END - SMOOTHNESS_RAMP_START), 0.0), 1.0)  # how much of the smoothness penalties apply

    def observe(self):
        """The drone's observation plus the tilt penalty weight (scaled to 0-1), so the policy can see how long it has been tilted."""
        obs = self.drone.get_observation(self.target, (self.tilt_weight / TILT_PENALTY_MAX_START,))
        return mirror_obs(obs) if self.mirrored else obs

    def time_to_go(self):
        """The least time the drone could need to reach the target and stop on it, if it could accelerate at TIME_OPTIMAL_ACCEL straight along the line to the
        target: accelerate (bang), then brake (bang), with the switch where the remaining distance equals the braking distance. Overshooting is brake, then come back."""
        shell = self.drone.shell
        to_x, to_y = self.target[0] - float(shell.center[0]), self.target[1] - float(shell.center[1])
        dist = math.hypot(to_x, to_y)
        vel_x, vel_y = (float(shell.vel[0, 0]) + float(shell.vel[1, 0])) / 2, (float(shell.vel[0, 1]) + float(shell.vel[1, 1])) / 2
        speed = (vel_x * to_x + vel_y * to_y) / (dist + 1e-8)  # velocity component toward the target (negative when moving away)
        accel = TIME_OPTIMAL_ACCEL
        if dist >= speed * abs(speed) / (2 * accel):  # not yet past the switching point
            peak = math.sqrt(accel * dist + speed * speed / 2)  # speed at the switch
            return (2 * peak - speed) / accel
        return speed / accel + 2 * math.sqrt((speed * speed / (2 * accel) - dist) / accel)  # will overshoot: brake to a stop, then return

    def spawn_target(self):
        """Uniform over the arena, except that far_fraction of the time the target must be at least FAR_TARGET_MIN_DIST from the drone
        (it takes the farthest of 20 tries if none is that far)."""
        drone_x, drone_y = float(self.drone.shell.center[0]), float(self.drone.shell.center[1])
        far = random.random() < self.far_fraction
        best, best_dist = None, -1.0
        for _ in range(20):
            candidate = (random.uniform(TARGET_MARGIN, ENV_WIDTH - TARGET_MARGIN), random.uniform(TARGET_MARGIN, ENV_HEIGHT - TARGET_MARGIN))
            dist = math.hypot(candidate[0] - drone_x, candidate[1] - drone_y)
            if not far or dist >= FAR_TARGET_MIN_DIST:
                best = candidate
                break
            if dist > best_dist:
                best, best_dist = candidate, dist
        self.target = np.array(best)

    def step(self, action):
        """Returns (obs, reward, terminated, truncated). terminated = left the screen, truncated = hit the step limit."""
        if self.mirrored:
            action = mirror_action(action)
        self.drone.apply_action(action, TIME_STEP_DELTA)
        self.drone.physics_move(GRAVITY, TIME_STEP_DELTA)
        self.steps += 1

        shell = self.drone.shell
        to_target_x, to_target_y = self.target[0] - float(shell.center[0]), self.target[1] - float(shell.center[1])
        dist = math.hypot(to_target_x, to_target_y)
        vel_x, vel_y = (float(shell.vel[0, 0]) + float(shell.vel[1, 0])) / 2, (float(shell.vel[0, 1]) + float(shell.vel[1, 1])) / 2
        reward = -TIME_PENALTY
        # Outwardness: positive when the left nozzle points bottom-left and the right nozzle points bottom-right
        left_out = self.drone.thruster_angles[0] - THRUSTER_NEUTRAL_ANGLE
        right_out = THRUSTER_NEUTRAL_ANGLE - self.drone.thruster_angles[1]
        reward += OUTWARD_REWARD * (max(-1.0, 1 - abs(left_out - OUTWARD_TARGET) / OUTWARD_TARGET) + max(-1.0, 1 - abs(right_out - OUTWARD_TARGET) / OUTWARD_TARGET)) / 2

        # Tilt penalty: the weight grows slowly while non-level and drops quickly once level
        angle = abs(self.drone.shell.angle)
        if angle < LEVEL_ANGLE:
            self.tilt_weight = max(TILT_PENALTY_MIN, self.tilt_weight * TILT_PENALTY_DECAY)
            self.level_weight = min(self.level_max, self.level_weight + self.level_max / LEVEL_REWARD_RAMP_STEPS)
            reward += self.level_weight * (1 - angle / LEVEL_ANGLE)  # level reward: grows with a long level streak
        else:
            self.tilt_weight = min(self.tilt_max, self.tilt_weight + self.tilt_max / TILT_PENALTY_RAMP_STEPS)
            self.level_weight *= LEVEL_REWARD_DECAY
        slow_factor = 1 + SLOW_TILT_EXTRA * max(0.0, 1 - self.drone.shell.linear_velocity_magnitude / SLOW_TILT_SPEED)
        reward -= self.tilt_weight * angle / np.pi * slow_factor
        reward -= ANGULAR_VELOCITY_PENALTY * max(0.0, abs(self.drone.shell.angular_velocity) - ANGULAR_VELOCITY_DEADBAND)

        # Smoothness penalties (phased in late in the session by smooth_scale)
        ang_vel = self.drone.shell.angular_velocity
        smoothness = ANGULAR_ACCEL_PENALTY * abs(ang_vel - self.prev_ang_vel)
        self.prev_ang_vel = ang_vel
        if self.prev_action is not None:
            smoothness += ACTION_CHANGE_PENALTY * float(np.abs(action - self.prev_action).mean())
        self.prev_action = action
        reward -= self.smooth_scale * smoothness

        reached = False
        calm = dist < REACH_DIST and self.drone.shell.linear_velocity_magnitude < self.reach_speed and abs(self.drone.shell.angular_velocity) < self.reach_ang_speed
        self.hold_steps = self.hold_steps + 1 if calm else 0
        if self.hold_steps >= self.reach_hold:
            self.hold_steps = 0
            reward += REACH_REWARD
            self.reaches += 1
            self.spawn_target()
            reached = True

        # Time-optimal shaping: pays the drop in the estimated time to reach the target (a finished target has none left). Undiscounted, so it telescopes and standing still far away earns nothing
        potential = 0.0 if reached else -self.time_to_go()
        reward += TIME_SHAPING_REWARD * (potential - self.potential)
        self.potential = -self.time_to_go() if reached else potential

        x, y = self.drone.shell.center
        terminated = not (0 <= x <= ENV_WIDTH and 0 <= y <= ENV_HEIGHT)
        if terminated:
            reward -= CRASH_PENALTY
        truncated = self.steps >= self.max_steps
        return self.observe(), reward, terminated, truncated


class RunningNorm:
    """Running mean and variance of everything it has seen. Used per feature for observations and as one scale for rewards."""

    def __init__(self, shape=()):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), 1e-4

    def update(self, x):  # x has shape [batch, *shape]
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        delta = x.mean(0) - self.mean
        total = self.count + n
        self.var = (self.var * self.count + x.var(0) * n + delta ** 2 * self.count * n / total) / total
        self.mean = self.mean + delta * n / total
        self.count = total

    def normalize(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -NORM_CLIP, NORM_CLIP)


class NumpyPolicy:
    """The actor's forward pass in plain numpy, so a worker process can act without torch."""

    def __init__(self, layers, log_std):
        self.layers, self.std = layers, np.exp(log_std)  # layers is a list of (weight, bias) pairs
        self.log_norm = float(np.sum(log_std)) + 0.5 * len(log_std) * math.log(2 * math.pi)

    def act(self, obs, rng):
        """Samples an action and returns it with its log-probability."""
        x = obs
        for weight, bias in self.layers[:-1]:
            x = np.tanh(weight @ x + bias)
        mean = self.layers[-1][0] @ x + self.layers[-1][1]
        noise = rng.standard_normal(len(mean))
        return mean + self.std * noise, -0.5 * float(noise @ noise) - self.log_norm


def symmetrize_obs_stats(norm):
    """With episodes mirrored half the time, observations are mirror symmetric on average, so the statistics should be too."""
    norm.mean = (norm.mean + norm.mean[OBS_MIRROR_PERM] * OBS_MIRROR_SIGN) / 2
    norm.var = (norm.var + norm.var[OBS_MIRROR_PERM]) / 2


def env_worker(conn, episode_length):
    """Runs one DroneEnv in its own process. On request it acts with the policy it was sent for a whole stretch of steps, entirely on its own."""
    env = DroneEnv(episode_length)
    rng = np.random.default_rng()
    obs, ep_return = env.reset(), 0.0
    while True:
        command, data = conn.recv()
        if command != "collect":
            break
        layers, log_std, obs_mean, obs_std, n_steps, progress = data
        env.set_ramp(progress)
        policy = NumpyPolicy(layers, log_std)
        raw_obs, acts = np.zeros((n_steps, OBS_DIM), dtype=np.float32), np.zeros((n_steps, ACT_DIM), dtype=np.float32)
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
        conn.send((raw_obs, acts, logp, rew, terminated, truncated, np.array(final_obs, dtype=np.float32).reshape(-1, OBS_DIM), obs, episodes, (ep_return, env.steps)))


class SubprocEnvs:
    """DroneEnvs in separate processes, so on separate CPU cores. Each runs a whole stretch of steps alone, so none waits on the others until its stretch is done."""

    def __init__(self, n_envs, episode_length):
        self.n_envs = n_envs
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=env_worker, args=(child, episode_length), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    def collect(self, layers, log_std, obs_mean, obs_std, n_steps, progress):
        """Every worker takes n_steps steps with the given policy and observation statistics, at this point (0 to 1) of the training session. Returns one result tuple per worker."""
        for conn in self.conns:
            conn.send(("collect", (layers, log_std, obs_mean, obs_std, n_steps, progress)))
        return [conn.recv() for conn in self.conns]

    def close(self):
        for conn in self.conns:
            conn.send(("close", None))
        for proc in self.procs:
            proc.join()


class EpisodeHandler:
    """Has every worker take steps_per_env steps with the current policy, then computes values, scaled rewards and advantages for the whole batch at once.
    An episode cut off by the epoch boundary continues next epoch."""

    def __init__(self, envs, actor, critic, obs_norm, ret_norm, steps_per_env):
        self.envs, self.actor, self.critic = envs, actor, critic
        self.obs_norm, self.ret_norm = obs_norm, ret_norm
        self.n_envs = envs.n_envs
        self.steps_per_env = steps_per_env
        self.disc_return = np.zeros(self.n_envs)  # discounted running return of each env, whose spread sets the reward scale
        self.partial = np.zeros((self.n_envs, 2))  # (return so far, steps so far) of each env's unfinished episode

    def unfinished(self):
        """Mean return and length so far of the episodes still in progress."""
        return tuple(self.partial.mean(axis=0))

    def _run_workers(self, n_steps, progress):
        layers = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in self.actor.net if isinstance(m, nn.Linear)]
        log_std = self.actor.log_std.detach().numpy().astype(np.float64)
        return self.envs.collect(layers, log_std, self.obs_norm.mean, np.sqrt(self.obs_norm.var + 1e-8), n_steps, progress)

    def warm_up(self, n_steps=128):
        """Takes a few steps just to get first observation statistics, so the first real epoch isn't unnormalized. The data is thrown away."""
        results = self._run_workers(n_steps, 0.0)
        self.obs_norm.update(np.concatenate([r[0] for r in results]))
        symmetrize_obs_stats(self.obs_norm)

    def collect(self, progress):
        T, N = self.steps_per_env, self.n_envs
        results = self._run_workers(T, progress)
        raw_obs, act, logp, raw_rew, terminated, truncated = (np.stack([r[i] for r in results], axis=1) for i in range(6))  # each [T, N, ...]
        last_obs = np.stack([r[7] for r in results])
        episodes = [episode for r in results for episode in r[8]]  # (return, length, reaches)
        self.partial = np.array([r[9] for r in results])
        done = terminated | truncated

        obs = self.obs_norm.normalize(raw_obs)  # with the same statistics the workers acted with
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            val = self.critic(to_tensor(obs.reshape(T * N, OBS_DIM))).numpy().reshape(T, N)
            next_val = np.zeros((T, N), dtype=np.float32)  # value of the state after each step
            next_val[:-1] = val[1:]
            next_val[-1] = self.critic(to_tensor(self.obs_norm.normalize(last_obs))).numpy()
            for n, r in enumerate(results):  # a step that ended an episode is followed by a restart, so use that episode's last observation instead
                ended = np.flatnonzero(done[:, n])
                if len(ended):
                    next_val[ended, n] = self.critic(to_tensor(self.obs_norm.normalize(r[6]))).numpy()
        next_val[terminated] = 0.0  # bootstrap on a timeout, not on a real end

        # Reward scale: the running std of each env's discounted return
        returns = np.zeros((T, N))
        for t in range(T):
            self.disc_return = self.disc_return * GAMMA + raw_rew[t]
            returns[t] = self.disc_return
            self.disc_return[done[t]] = 0.0
        self.ret_norm.update(returns.reshape(-1))
        rew = np.clip(raw_rew / np.sqrt(self.ret_norm.var + 1e-8), -NORM_CLIP, NORM_CLIP).astype(np.float32)
        self.obs_norm.update(raw_obs.reshape(T * N, OBS_DIM))
        symmetrize_obs_stats(self.obs_norm)

        # GAE (generalized advantage estimation) for all envs at once, restarting the running sum at each episode boundary
        adv = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            delta = rew[t] + GAMMA * next_val[t] - val[t]
            last = delta + GAMMA * GAE_LAMBDA * (1 - done[t]) * last
            adv[t] = last

        batch = {"obs": obs.astype(np.float32), "act": act, "logp": logp, "adv": adv, "ret": adv + val}
        return {k: torch.as_tensor(v.reshape(T * N, *v.shape[2:])) for k, v in batch.items()}, episodes


class TrainHandler:
    def __init__(self, run_dir, hidden_sizes=HIDDEN_SIZES, epochs=EPOCHS, episode_length=MAX_EPISODE_STEPS, steps_per_epoch=STEPS_PER_EPOCH, num_envs=NUM_ENVS, resume_dir=None, ramp_epochs=None, lr_decay_end=None, lr_schedule=None):
        self.run_dir = run_dir
        self.lr_decay_end = LR_DECAY_END if lr_decay_end is None else lr_decay_end
        self.lr_schedule = lr_schedule or LR_SCHEDULE
        self.ramp_epochs = ramp_epochs or epochs  # how many epochs the curricula (rewards, targets, smoothness) take to play out
        self.rows_written = {"epochs": 0, "episodes": 0}
        self.batch_size = steps_per_epoch * num_envs
        self.epochs = epochs
        self.start_time = time.time()
        self.actor, self.critic = Actor(hidden_sizes), Critic(hidden_sizes)
        if resume_dir is not None:  # continue from a previous run's weights (the optimizer's momentum is not saved, so it starts fresh)
            self.actor.load_state_dict(torch.load(resume_dir / "actor.pt"))
            self.critic.load_state_dict(torch.load(resume_dir / "critic.pt"))
        self.base_lrs = [PI_LR, VF_LR]
        self.optimizer = torch.optim.Adam([{"params": self.actor.parameters(), "lr": PI_LR}, {"params": self.critic.parameters(), "lr": VF_LR}])
        self.obs_norm, self.ret_norm = RunningNorm((OBS_DIM,)), RunningNorm()
        if resume_dir is not None:
            norms = np.load(resume_dir / "norm.npz")
            self.obs_norm.mean, self.obs_norm.var, self.obs_norm.count = norms["obs_mean"], norms["obs_var"], float(norms["obs_count"])
            self.ret_norm.mean, self.ret_norm.var, self.ret_norm.count = norms["ret_mean"], norms["ret_var"], float(norms["ret_count"])
            symmetrize_obs_stats(self.obs_norm)
        envs = SubprocEnvs(num_envs, episode_length)
        self.episode_handler = EpisodeHandler(envs, self.actor, self.critic, self.obs_norm, self.ret_norm, steps_per_epoch)
        if resume_dir is None:
            self.episode_handler.warm_up()
        self.epoch_rows = []
        self.episode_rows = []

    def run_epoch(self, epoch):
        progress = (epoch - 1) / max(self.epochs - 1, 1)  # 0 at the first epoch, 1 at the last
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            factor = 1 - (1 - self.lr_decay_end) * progress if self.lr_schedule == "linear" else self.lr_decay_end ** progress
            group["lr"] = base_lr * factor
        ramp_progress = min((epoch - 1) / max(self.ramp_epochs - 1, 1), 1.0)  # drives the curricula
        batch, episodes = self.episode_handler.collect(ramp_progress)
        pi_loss, v_loss = self.backprop(batch)
        self.log_results(epoch, episodes, pi_loss, v_loss)

    def backprop(self, batch):
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        for _ in range(UPDATE_EPOCHS):
            for idx in torch.randperm(self.batch_size).split(self.batch_size // NUM_MINIBATCHES):
                logp = self.actor.dist(batch["obs"][idx]).log_prob(batch["act"][idx]).sum(-1)
                ratio = (logp - batch["logp"][idx]).exp()
                pi_loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - CLIP_RATIO, 1 + CLIP_RATIO) * adv[idx]).mean()
                v_loss = ((self.critic(batch["obs"][idx]) - batch["ret"][idx]) ** 2).mean()
                self.optimizer.zero_grad()
                (pi_loss + 0.5 * v_loss).backward()  # actor and critic share no weights, so one combined loss is equivalent to two separate steps
                self.optimizer.step()
        return pi_loss.item(), v_loss.item()

    def log_results(self, epoch, episodes, pi_loss, v_loss):
        row = {"epoch": epoch, "episodes": len(episodes),
               "mean_return": np.mean([e[0] for e in episodes]) if episodes else float("nan"),
               "mean_length": np.mean([e[1] for e in episodes]) if episodes else float("nan"),
               "total_reaches": sum(e[2] for e in episodes), "pi_loss": pi_loss, "v_loss": v_loss,
               "actor_lr": self.optimizer.param_groups[0]["lr"], "critic_lr": self.optimizer.param_groups[1]["lr"]}
        self.epoch_rows.append(row)
        self.episode_rows += [{"epoch": epoch, "return": r, "length": length, "reaches": k} for r, length, k in episodes]
        # When no episode finished this epoch, print the unfinished episode's return so far (marked *); the CSV and plot keep nan
        partial = "" if episodes else "*"
        shown_return, shown_length = (row["mean_return"], row["mean_length"]) if episodes else self.episode_handler.unfinished()
        print(f"epoch {epoch:3d}/{self.epochs} | {time.time() - self.start_time:5.0f}s | episodes {row['episodes']:2d} | mean return {shown_return:8.2f}{partial or ' '}| mean length {shown_length:6.1f}{partial or ' '}| reaches {row['total_reaches']:2d} | pi {pi_loss:7.4f} | v {v_loss:8.4f} | lr {row['actor_lr']:.1e}/{row['critic_lr']:.1e}", flush=True)

    def train(self):
        try:
            for epoch in range(1, self.epochs + 1):
                self.run_epoch(epoch)
                if epoch % SAVE_EVERY == 0 and epoch < self.epochs:
                    self.save()
                    if epoch % SNAPSHOT_EVERY == 0:
                        self.save_models(self.run_dir / "snapshots" / f"epoch{epoch}")
                        print(f"saved checkpoint and a snapshot of the model at epoch {epoch}", flush=True)
        finally:
            self.episode_handler.envs.close()
        self.save(show_plot=True)

    def save_models(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), directory / "actor.pt")
        torch.save(self.critic.state_dict(), directory / "critic.pt")
        np.savez(directory / "norm.npz", obs_mean=self.obs_norm.mean, obs_var=self.obs_norm.var, obs_count=self.obs_norm.count,
                 ret_mean=self.ret_norm.mean, ret_var=self.ret_norm.var, ret_count=self.ret_norm.count)

    def save(self, show_plot=False):
        self.save_models(self.run_dir)
        fields = {"epochs": ["epoch", "episodes", "mean_return", "mean_length", "total_reaches", "pi_loss", "v_loss", "actor_lr", "critic_lr"],
                  "episodes": ["epoch", "return", "length", "reaches"]}
        for name, rows in (("epochs", self.epoch_rows), ("episodes", self.episode_rows)):  # only the rows added since the last save are written
            first = self.rows_written[name] == 0
            with open(self.run_dir / f"{name}.csv", "w" if first else "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields[name])
                if first:
                    writer.writeheader()
                writer.writerows(rows[self.rows_written[name]:])
            self.rows_written[name] = len(rows)

        plt.figure()
        plt.plot([r["epoch"] for r in self.epoch_rows], [r["mean_return"] for r in self.epoch_rows])
        plt.xlabel("epoch")
        plt.ylabel("mean episode return")
        plt.title(self.run_dir.name)
        plt.savefig(self.run_dir / "rewards.png")
        if show_plot:
            plt.show()
        plt.close()


RUNS_DIR = Path(__file__).parent / "runs"


def new_run_dir(resume_name=None):
    """runs/run<N> for a fresh run, or runs/<resume_name>.<k> (e.g. run2.1) when continuing an existing one."""
    if resume_name is None:
        existing = [int(m.group(1)) for p in RUNS_DIR.glob("run*") if (m := re.fullmatch(r"run(\d+)", p.name))]
        name = f"run{max(existing, default=0) + 1}"
    else:
        k = 1
        while (RUNS_DIR / f"{resume_name}.{k}").exists():
            k += 1
        name = f"{resume_name}.{k}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True)
    return run_dir


def parse_run(text):
    """Accepts '2', 'run2', '2.1' or 'run2.1' and returns the run folder name, which must hold a saved actor and critic."""
    name = "run" + re.sub(r"^run", "", text)
    if not re.fullmatch(r"run\d+(\.\d+)*", name) or not all((RUNS_DIR / name / f).exists() for f in ("actor.pt", "critic.pt")):
        raise argparse.ArgumentTypeError(f"no saved run {text!r} (looked for actor.pt and critic.pt in runs/{name}/)")
    return name


def hidden_sizes_of(run_name):
    """Reads the hidden layer sizes from a saved actor, checking it matches the current observation and action sizes."""
    state = torch.load(RUNS_DIR / run_name / "actor.pt")
    layers = [w.shape for k, w in state.items() if k.startswith("net.") and k.endswith("weight")]
    if layers[0][1] != OBS_DIM or layers[-1][0] != ACT_DIM:
        raise argparse.ArgumentTypeError(f"{run_name} has {layers[0][1]} inputs and {layers[-1][0]} outputs, but the current setup uses {OBS_DIM} and {ACT_DIM}")
    return [shape[0] for shape in layers[:-1]]


def parse_model(text):
    """Parses '[16, 16, 8, 2]' into [16, 16, 8, 2]; the first value must be the observation size and the last the action size."""
    try:
        sizes = json.loads(text)
        assert isinstance(sizes, list) and len(sizes) >= 2 and all(isinstance(n, int) and n > 0 for n in sizes)
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(f'expected an array like "[{OBS_DIM}, 16, 8, {ACT_DIM}]", got {text!r}')
    if sizes[0] != OBS_DIM or sizes[-1] != ACT_DIM:
        raise argparse.ArgumentTypeError(f"the first size must be {OBS_DIM} (observations) and the last {ACT_DIM} (actions), got {sizes}")
    return sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the drone with PPO. Results are saved to runs/run<N>/.")
    parser.add_argument("--model", type=parse_model, default=None, help=f'network sizes as an array, e.g. "[{OBS_DIM}, 16, 8, {ACT_DIM}]" (default: [{OBS_DIM}, {", ".join(map(str, HIDDEN_SIZES))}, {ACT_DIM}]; ignored with --resume)')
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="number of epochs (default: %(default)s)")
    parser.add_argument("--lr-decay", type=float, default=None, help=f"fraction of the starting learning rates reached by the last epoch, exponentially (default: {LR_DECAY_END})")
    parser.add_argument("--lr-schedule", choices=["linear", "exponential"], default=None, help=f"how the learning rates fall to the --lr-decay fraction (default: {LR_SCHEDULE})")
    parser.add_argument("--ramp-epochs", type=int, default=None, help="epochs over which the reward, target and smoothness curricula play out (default: --epochs). For a very long session set this lower so the curricula don't stretch")
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH, help="steps each environment takes per epoch; the epoch's total is this times --num-envs (default: %(default)s)")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="environments run together, each in its own process (default: %(default)s)")
    parser.add_argument("--episode-length", type=int, default=MAX_EPISODE_STEPS, help="max steps per episode (default: %(default)s)")
    parser.add_argument("--resume", type=parse_run, default=None, metavar="RUN", help="continue training a saved run, e.g. 2 or 2.1; results go to a new folder such as runs/run2.1")
    args = parser.parse_args()

    if args.resume:
        if not (RUNS_DIR / args.resume / "norm.npz").exists():
            parser.error(f"{args.resume} has no saved normalization statistics (it was trained before observation normalization was added), so it can't be continued")
        if args.model:
            parser.error("--model can't be combined with --resume; the model size comes from the saved run")
        try:
            hidden_sizes = hidden_sizes_of(args.resume)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))
        model = [OBS_DIM, *hidden_sizes, ACT_DIM]
    else:
        model = args.model or [OBS_DIM, *HIDDEN_SIZES, ACT_DIM]
        hidden_sizes = model[1:-1]

    run_dir = new_run_dir(args.resume)
    print(f"run folder: {run_dir} | model {model} | {args.epochs} epochs x {args.steps_per_epoch * args.num_envs} steps ({args.num_envs} envs x {args.steps_per_epoch}) | episode length {args.episode_length}" + (f" | curricula over {args.ramp_epochs} epochs" if args.ramp_epochs else "") + (f" | continuing {args.resume}" if args.resume else ""), flush=True)
    print(f"reward ramp: level cap {LEVEL_REWARD_MAX_START} -> {LEVEL_REWARD_MAX_END} and tilt cap {TILT_PENALTY_MAX_START} -> {TILT_PENALTY_MAX_END}, reached {REWARD_RAMP_END:.0%} of the way through the session; reach limits {REACH_SPEED_START:.0f}->{REACH_SPEED_END:.0f} px/s and {REACH_ANGULAR_SPEED_START}->{REACH_ANGULAR_SPEED_END} rad/s over the same ramp; smoothness penalties phase in from {SMOOTHNESS_RAMP_START:.0%} to {SMOOTHNESS_RAMP_END:.0%}; reach hold {REACH_HOLD_STEPS_START}->{REACH_HOLD_STEPS_END} steps; far targets (at least {FAR_TARGET_MIN_DIST:.0f} px) grow from 0% to {FAR_TARGET_FRACTION_END:.0%} of new targets by {TARGET_CURRICULUM_END:.0%}", flush=True)
    print("* = no episode finished this epoch, so the mean so far of the unfinished episodes is shown", flush=True)
    TrainHandler(run_dir, hidden_sizes, args.epochs, args.episode_length, args.steps_per_epoch, args.num_envs, RUNS_DIR / args.resume if args.resume else None, args.ramp_epochs, args.lr_decay, args.lr_schedule).train()
