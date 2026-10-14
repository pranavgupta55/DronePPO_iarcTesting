"""Quiet, spec-driven PPO trainer derived from ppo.py. train(spec, run_dir) trains one session and saves actor.pt, critic.pt, norm.npz,
epochs.csv, spec.json and done.json into run_dir."""
import csv
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from drone_rl.config import OBS_BASE_DIM, ACT_DIM
from drone_rl.env import NORM_CLIP, DEFAULTS, make_spec, load_scheme, mirror_tools, env_worker


def mlp(in_dim, out_dim, hidden_sizes):
    sizes = [in_dim, *hidden_sizes]
    layers = []
    for n_in, n_out in zip(sizes, sizes[1:]):
        layers += [nn.Linear(n_in, n_out), nn.Tanh()]
    return nn.Sequential(*layers, nn.Linear(sizes[-1], out_dim))


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, init_log_std):
        super().__init__()
        self.net = mlp(obs_dim, ACT_DIM, hidden_sizes)
        self.log_std = nn.Parameter(torch.tensor([float(x) for x in init_log_std]) if isinstance(init_log_std, (list, tuple)) else torch.full((ACT_DIM,), float(init_log_std)))  # one value, or one per action: force 1, force 2, angle 1, angle 2

    def dist(self, obs):
        return Normal(self.net(obs), self.log_std.exp())


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden_sizes)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


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


class SubprocEnvs:
    """DroneEnvs in separate processes. Each runs a whole stretch of steps alone, so none waits on the others until its stretch is done."""

    def __init__(self, spec):
        self.n_envs = spec["num_envs"]
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for rank in range(self.n_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=env_worker, args=(child, spec, rank), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    def collect(self, layers, log_std, obs_mean, obs_std, n_steps, progress):
        for conn in self.conns:
            conn.send(("collect", (layers, log_std, obs_mean, obs_std, n_steps, progress)))
        results = [conn.recv() for conn in self.conns]
        for r in results:
            if isinstance(r[0], str):
                raise RuntimeError(f"env worker failed:\n{r[1]}")
        return results

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


class EpisodeHandler:
    """Has every worker take steps_per_env steps with the current policy, then computes values, scaled rewards and advantages for the whole batch.
    An episode cut off by the epoch boundary continues next epoch."""

    def __init__(self, spec, envs, actor, critic, obs_norm, ret_norm, obs_dim, mirror_perm, mirror_sign):
        self.spec, self.envs, self.actor, self.critic = spec, envs, actor, critic
        self.obs_norm, self.ret_norm, self.obs_dim = obs_norm, ret_norm, obs_dim
        self.mirror_perm, self.mirror_sign = mirror_perm, mirror_sign
        self.n_envs = envs.n_envs
        self.steps_per_env = spec["steps_per_epoch"]
        self.gamma, self.lam = spec["gamma"], spec["gae_lambda"]
        self.disc_return = np.zeros(self.n_envs)  # discounted running return of each env, whose spread sets the reward scale

    def symmetrize(self):
        """With episodes mirrored half the time, observations are mirror symmetric on average, so the statistics should be too."""
        if self.spec["mirror"]:
            n = self.obs_norm
            n.mean = (n.mean + n.mean[self.mirror_perm] * self.mirror_sign) / 2
            n.var = (n.var + n.var[self.mirror_perm]) / 2

    def _run_workers(self, n_steps, progress):
        layers = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in self.actor.net if isinstance(m, nn.Linear)]
        log_std = self.actor.log_std.detach().numpy().astype(np.float64)
        if self.spec["obs_norm"]:
            mean, std = self.obs_norm.mean, np.sqrt(self.obs_norm.var + 1e-8)
        else:
            mean, std = np.zeros(self.obs_dim), np.ones(self.obs_dim)
        return self.envs.collect(layers, log_std, mean, std, n_steps, progress)

    def warm_up(self, n_steps=128):
        """A few steps just to get first observation statistics. The data is thrown away."""
        results = self._run_workers(n_steps, 0.0)
        self.obs_norm.update(np.concatenate([r[0] for r in results]))
        self.symmetrize()

    def _norm_obs(self, x):
        return self.obs_norm.normalize(x) if self.spec["obs_norm"] else np.asarray(x)

    def collect(self, progress):
        T, N = self.steps_per_env, self.n_envs
        results = self._run_workers(T, progress)
        raw_obs, act, logp, raw_rew, terminated, truncated = (np.stack([r[i] for r in results], axis=1) for i in range(6))  # each [T, N, ...]
        if not (np.isfinite(raw_rew).all() and np.isfinite(raw_obs).all()):
            raise FloatingPointError("NaN/inf in rewards or observations")
        last_obs = np.stack([r[7] for r in results])
        episodes = [episode for r in results for episode in r[8]]  # (return, length, reaches)
        done = terminated | truncated

        obs = self._norm_obs(raw_obs)  # with the same statistics the workers acted with
        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            val = self.critic(to_tensor(obs.reshape(T * N, self.obs_dim))).numpy().reshape(T, N)
            next_val = np.zeros((T, N), dtype=np.float32)  # value of the state after each step
            next_val[:-1] = val[1:]
            next_val[-1] = self.critic(to_tensor(self._norm_obs(last_obs))).numpy()
            for n, r in enumerate(results):  # a step that ended an episode is followed by a restart, so use that episode's last observation instead
                ended = np.flatnonzero(done[:, n])
                if len(ended):
                    next_val[ended, n] = self.critic(to_tensor(self._norm_obs(r[6]))).numpy()
        next_val[terminated] = 0.0  # bootstrap on a timeout, not on a real end

        # Reward scale: the running std of each env's discounted return
        if self.spec["reward_scaling"]:
            returns = np.zeros((T, N))
            for t in range(T):
                self.disc_return = self.disc_return * self.gamma + raw_rew[t]
                returns[t] = self.disc_return
                self.disc_return[done[t]] = 0.0
            self.ret_norm.update(returns.reshape(-1))
            rew = np.clip(raw_rew / np.sqrt(self.ret_norm.var + 1e-8), -NORM_CLIP, NORM_CLIP).astype(np.float32)
        else:
            rew = raw_rew.astype(np.float32)
        if self.spec["obs_norm"] and not self.spec["freeze_obs_norm"]:  # frozen when a pretrained network expects exactly its own observation scaling
            self.obs_norm.update(raw_obs.reshape(T * N, self.obs_dim))
            self.symmetrize()

        # GAE (generalized advantage estimation) for all envs at once, restarting the running sum at each episode boundary
        adv = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            delta = rew[t] + self.gamma * next_val[t] - val[t]
            last = delta + self.gamma * self.lam * (1 - done[t]) * last
            adv[t] = last

        batch = {"obs": obs.astype(np.float32), "act": act, "logp": logp, "adv": adv, "ret": adv + val}
        return {k: torch.as_tensor(v.reshape(T * N, *v.shape[2:])) for k, v in batch.items()}, episodes


class TrainHandler:
    def __init__(self, spec, run_dir):
        self.spec, self.run_dir = spec, Path(run_dir)
        module, _ = load_scheme(spec["scheme"], spec["params"])  # also validates the scheme before any worker starts
        self.obs_dim = OBS_BASE_DIM + int(module.EXTRA_OBS)
        mirror_perm, mirror_sign = mirror_tools(module.EXTRA_OBS_MIRROR)
        self.epochs = spec["epochs"]
        self.start_time = time.time()
        self.batch_size = spec["steps_per_epoch"] * spec["num_envs"]
        hidden = spec["hidden"]
        self.actor, self.critic = Actor(self.obs_dim, hidden, spec["init_log_std"]), Critic(self.obs_dim, hidden)
        self.base_lrs = [spec["pi_lr"], spec["vf_lr"]]
        self.optimizer = torch.optim.Adam([{"params": self.actor.parameters(), "lr": self.base_lrs[0]}, {"params": self.critic.parameters(), "lr": self.base_lrs[1]}])
        self.obs_norm, self.ret_norm = RunningNorm((self.obs_dim,)), RunningNorm()
        if spec["init_from"]:  # start from a pretrained actor and its observation statistics (the critic starts fresh)
            source = Path(spec["init_from"])
            if not (source / "actor.pt").exists():  # a run id or name instead of a path
                from drone_rl import registry
                source = registry.find_run(spec["init_from"])
            state = torch.load(source / "actor.pt", map_location="cpu")
            self.actor.net.load_state_dict({k[len("net."):]: v for k, v in state.items() if k.startswith("net.")})
            norms = np.load(source / "norm.npz")
            self.obs_norm.mean, self.obs_norm.var, self.obs_norm.count = norms["obs_mean"], norms["obs_var"], float(norms["obs_count"])
        self.envs = SubprocEnvs(spec)
        self.episode_handler = EpisodeHandler(spec, self.envs, self.actor, self.critic, self.obs_norm, self.ret_norm, self.obs_dim, mirror_perm, mirror_sign)
        self.rows = []

    def run_epoch(self, epoch):
        s = self.spec
        progress = (epoch - 1) / max(self.epochs - 1, 1)  # 0 at the first epoch, 1 at the last
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            factor = 1 - (1 - s["lr_decay_end"]) * progress if s["lr_schedule"] == "linear" else s["lr_decay_end"] ** progress
            group["lr"] = base_lr * factor
        if epoch <= s["critic_warmup_epochs"]:
            self.optimizer.param_groups[0]["lr"] = 0.0  # actor frozen while the new critic catches up
        batch, episodes = self.episode_handler.collect(progress)
        pi_loss, v_loss = self.backprop(batch)
        row = {"epoch": epoch, "time": round(time.time() - self.start_time, 2), "episodes": len(episodes),
               "mean_return": float(np.mean([e[0] for e in episodes])) if episodes else float("nan"),
               "mean_length": float(np.mean([e[1] for e in episodes])) if episodes else float("nan"),
               "total_reaches": sum(e[2] for e in episodes), "pi_loss": pi_loss, "v_loss": v_loss, "actor_lr": self.optimizer.param_groups[0]["lr"]}
        self.rows.append(row)
        if epoch % 25 == 0 or epoch == self.epochs:
            print(f"epoch {epoch:3d}/{self.epochs} {row['time']:5.0f}s | eps {row['episodes']:2d} | ret {row['mean_return']:9.1f} | len {row['mean_length']:6.0f} | reaches {row['total_reaches']:3d} | pi {pi_loss:.4f} | v {v_loss:.4f}", flush=True)

    def backprop(self, batch):
        s = self.spec
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        for _ in range(s["update_epochs"]):
            for idx in torch.randperm(self.batch_size).split(max(self.batch_size // s["num_minibatches"], 1)):
                logp = self.actor.dist(batch["obs"][idx]).log_prob(batch["act"][idx]).sum(-1)
                ratio = (logp - batch["logp"][idx]).exp()
                pi_loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - s["clip"], 1 + s["clip"]) * adv[idx]).mean()
                v_loss = ((self.critic(batch["obs"][idx]) - batch["ret"][idx]) ** 2).mean()
                self.optimizer.zero_grad()
                (pi_loss + 0.5 * v_loss).backward()  # actor and critic share no weights, so one combined loss is equivalent to two separate steps
                self.optimizer.step()
        if not (np.isfinite(pi_loss.item()) and np.isfinite(v_loss.item())):
            raise FloatingPointError("NaN/inf loss")
        return pi_loss.item(), v_loss.item()

    def train(self):
        try:
            if self.spec["obs_norm"] and not self.spec["init_from"]:
                self.episode_handler.warm_up()
            for epoch in range(1, self.epochs + 1):
                self.run_epoch(epoch)
        finally:
            self.envs.close()
        self.save()
        return self.finish()

    def save(self):
        d = self.run_dir
        torch.save(self.actor.state_dict(), d / "actor.pt")
        torch.save(self.critic.state_dict(), d / "critic.pt")
        np.savez(d / "norm.npz", obs_mean=self.obs_norm.mean, obs_var=self.obs_norm.var, obs_count=self.obs_norm.count,
                 ret_mean=self.ret_norm.mean, ret_var=self.ret_norm.var, ret_count=self.ret_norm.count)
        with open(d / "epochs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def finish(self):
        tail = self.rows[-10:]
        mean = lambda key: float(np.nanmean([r[key] for r in tail])) if not np.all(np.isnan([r[key] for r in tail])) else None
        done = {"epochs": self.epochs, "wall_seconds": round(time.time() - self.start_time, 1), "last": self.rows[-1],
                "final_return": mean("mean_return"), "final_length": mean("mean_length"), "final_reaches_per_epoch": mean("total_reaches")}
        (self.run_dir / "done.json").write_text(json.dumps(done, indent=1))
        return done


def train(spec, run_dir):
    spec = make_spec(spec)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=1))
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    torch.set_num_threads(int(spec["torch_threads"]))
    return TrainHandler(spec, run_dir).train()
