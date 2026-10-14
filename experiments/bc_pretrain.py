"""Behavior cloning of the bang-bang controller into the fixed [17+extra, 16, 8, 4] network (standard supervised learning), with DAgger rounds.
python3 bc_pretrain.py --name bc_feat --scheme bangbang_features --params '{"approach_w":40}' --samples 400000 --dagger 3
Round 0: the controller flies (with random start states, random retargets and action noise) and every (observation, controller action) pair is recorded.
Rounds 1..N: the student flies (mixed with the controller with probability beta) and the controller labels the states the student reaches; all data is kept.
Saves runs/bc/<name>/{actor.pt, norm.npz, spec.json} in the layout the PPO trainer (spec "init_from") and evaluate.py read."""
import argparse
import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np

from bangbang_controller import control, DEFAULTS as BB
from core.config import ENV_WIDTH, ENV_HEIGHT, ACT_DIM, THRUSTER_NEUTRAL_ANGLE
from core.env import DroneEnv, NumpyPolicy, make_spec, load_scheme, NORM_CLIP

LABEL_CLIP = 3.0
MILD_FRACTION = 0.8
FORCE_RATE = 0.5  # action 1 = 0.5 thruster force per step, so the controller's biggest force jumps (0 -> 1) are actions of 2


def randomize_state(env, rng):
    """A varied start state so the data covers more than the spawn: anywhere in the world, any speed, tilt and spin, any thruster forces."""
    d = env.drone
    pos = np.array([rng.uniform(80, ENV_WIDTH - 80), rng.uniform(80, ENV_HEIGHT - 80)])
    big = rng.random() < 0.6
    vel = rng.uniform(-700, 700, 2) if big else rng.uniform(-60, 60, 2)
    angle = rng.uniform(-math.pi, math.pi) if rng.random() < 0.35 else rng.uniform(-0.4, 0.4)
    spin = rng.uniform(-10, 10) if rng.random() < 0.3 else rng.uniform(-1, 1)
    d.shell.set_physical_state(pos, angle, vel.astype(np.float32), spin)
    if rng.random() < 0.5:
        d.thruster_forces = [rng.uniform(0, 1), rng.uniform(0, 1)]
    for key in ("bb_braking", "bb_unwound"):
        if hasattr(env, key):
            delattr(env, key)


def collect(job):
    """Runs episodes until n_steps are recorded. student = None (the controller flies) or (layers, obs_mean, obs_std, beta): with probability beta per step the controller's action is executed instead."""
    spec, seed, n_steps, student = job
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    module, scheme = load_scheme(spec["scheme"], spec["params"])
    env = DroneEnv(spec, module, scheme, max_steps=10 ** 9)
    env.set_ramp(1.0)
    env.set_eval_mode()
    policy = NumpyPolicy(student[0], np.zeros(ACT_DIM)) if student else None
    obs_list, lab_list = [], []
    reaches = crashes = episodes = 0
    while len(obs_list) < n_steps:
        obs = env.reset()
        if student is None or rng.random() >= MILD_FRACTION:  # student rollouts mostly start from the normal spawn: wild states are a small part of what it meets
            randomize_state(env, rng)
        else:
            for key in ("bb_braking", "bb_unwound"):
                if hasattr(env, key):
                    delattr(env, key)
        obs = env.observe()
        noise_sd = rng.choice([0.0, 0.0, 0.1, 0.3])
        length = int(rng.integers(300, 1500))
        episodes += 1
        for t in range(length):
            env.bb_braking = getattr(env, "bb_braking", False)
            label = np.clip(control(env, BB), -LABEL_CLIP, LABEL_CLIP)
            obs_list.append(obs.copy())
            lab_list.append(label)
            if policy is not None and rng.random() >= student[3]:
                act = np.clip(policy.mean(np.clip((obs - student[1]) / student[2], -NORM_CLIP, NORM_CLIP)), -LABEL_CLIP, LABEL_CLIP)
            else:
                act = label
            act = act + noise_sd * rng.standard_normal(ACT_DIM)
            if rng.random() < 1 / 300:  # the target jumps now and then, like a moving mouse
                env.target = np.array([rng.uniform(10, ENV_WIDTH - 10), rng.uniform(10, ENV_HEIGHT - 10)])
            r0 = env.reaches
            obs, _, term, trunc = env.step(act)
            reaches += env.reaches - r0
            if term:
                crashes += 1
                break
            if len(obs_list) >= n_steps:
                break
    return np.array(obs_list, dtype=np.float32), np.array(lab_list, dtype=np.float32), reaches, crashes, episodes


def collect_parallel(pool, spec, n_steps, student, seed0, procs):
    jobs = [(spec, seed0 + i, n_steps // procs, student) for i in range(procs)]
    out = pool.map(collect, jobs, chunksize=1)
    return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]), sum(o[2] for o in out), sum(o[3] for o in out), sum(o[4] for o in out))


def closed_loop(job):
    """Mean-action rollout in the fixed evaluation mode; returns (reaches, crashes, steps) over the given episodes."""
    spec, layers, obs_mean, obs_std, seeds, steps = job
    module, scheme = load_scheme(spec["scheme"], spec["params"])
    policy = NumpyPolicy(layers, np.zeros(ACT_DIM))
    reaches = crashes = 0
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        env = DroneEnv(spec, module, scheme, max_steps=steps)
        env.set_ramp(1.0)
        env.set_eval_mode()
        obs = env.reset()
        for _ in range(steps):
            obs, _, term, trunc = env.step(policy.mean(np.clip((obs - obs_mean) / obs_std, -NORM_CLIP, NORM_CLIP)))
            if term:
                crashes += 1
            if term or trunc:
                break
        reaches += env.reaches
    return reaches, crashes, len(seeds)


def fit(model, obs, labels, obs_mean, obs_std, epochs, lr, batch=4096, log=print):
    import torch
    x = torch.as_tensor(np.clip((obs - obs_mean) / obs_std, -NORM_CLIP, NORM_CLIP), dtype=torch.float32)
    y = torch.as_tensor(labels)
    order = torch.randperm(len(x))
    n_val = max(len(x) // 20, 1)
    val, train = order[:n_val], order[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(len(train) / batch), eta_min=lr * 0.02)
    for epoch in range(1, epochs + 1):
        perm = train[torch.randperm(len(train))]
        for idx in perm.split(batch):
            loss = torch.nn.functional.smooth_l1_loss(model(x[idx]), y[idx], beta=0.3)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(model(x[val]), y[val], reduction="none").mean(0)
            log(f"    epoch {epoch:3d} val mse per output {[round(float(a), 4) for a in v]}")


def main():
    import torch
    from core.trainer import mlp
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scheme", default="bangbang_features")
    ap.add_argument("--params", default='{"approach_w": 40}')
    ap.add_argument("--samples", type=int, default=400000)
    ap.add_argument("--dagger", type=int, default=3)
    ap.add_argument("--dagger_samples", type=int, default=200000)
    ap.add_argument("--epochs0", type=int, default=60)
    ap.add_argument("--epochs_dagger", type=int, default=30)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--blurb", required=True, help='"short name\\nwhat it tests, what changed from the run it branched from"')
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    spec = make_spec({"name": a.name, "scheme": a.scheme, "params": json.loads(a.params), "force_rate": FORCE_RATE, "mirror": False, "num_envs": 2,
                      "init_log_std": -2.5, "target_margin": 10.0})
    out = Path(__file__).parent / "runs" / "bc" / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    (out / "blurb.txt").write_text(a.blurb.replace("\\n", "\n").strip() + "\n")
    t0 = time.time()
    log = lambda *m: print(f"[{time.time() - t0:5.0f}s]", *m, flush=True)
    module, _ = load_scheme(spec["scheme"], spec["params"])
    obs_dim = 17 + int(module.EXTRA_OBS)
    model = mlp(obs_dim, ACT_DIM, spec["hidden"])
    best_score = -1.0
    with mp.get_context("spawn").Pool(a.procs) as pool:
        obs, lab, reaches, crashes, episodes = collect_parallel(pool, spec, a.samples, None, 1000, a.procs)
        log(f"round 0: controller data {len(obs)} samples, {episodes} episodes, {reaches} reaches, {crashes} crashes")
        obs_mean, obs_var = obs.mean(0).astype(np.float64), obs.var(0).astype(np.float64)
        obs_std = np.sqrt(obs_var + 1e-8)
        log(f"  label std per output {lab.std(0).round(3)} (mean {lab.mean(0).round(3)})")
        for rnd in range(a.dagger + 1):
            fit(model, obs, lab, obs_mean, obs_std, a.epochs0 if rnd == 0 else a.epochs_dagger, a.lr if rnd == 0 else a.lr / 3, log=log)
            layers = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)) for m in model if isinstance(m, torch.nn.Linear)]
            n_eval = 24
            res = pool.map(closed_loop, [(spec, layers, obs_mean, obs_std, list(range(5000 + i * n_eval // a.procs, 5000 + (i + 1) * n_eval // a.procs)), 2048) for i in range(a.procs)])
            log(f"round {rnd}: closed loop reaches/ep {sum(r[0] for r in res) / n_eval:.2f}, crash {sum(r[1] for r in res) / n_eval:.0%} ({len(obs)} samples)")
            score = sum(r[0] for r in res) / n_eval * (1 - sum(r[1] for r in res) / n_eval)  # reaches per episode, discounted by the crash share
            if score > best_score:  # keep the best round, not the last: more DAgger data does not always help a network this small
                best_score = score
                torch.save({**{f"net.{k}": v for k, v in model.state_dict().items()}, "log_std": torch.full((ACT_DIM,), spec["init_log_std"])}, out / "actor.pt")
                np.savez(out / "norm.npz", obs_mean=obs_mean, obs_var=obs_var, obs_count=100000.0)
                log(f"  saved as best (score {score:.2f})")
            if rnd == a.dagger:
                break
            beta = max(0.0, 0.5 - 0.15 * rnd)  # share of controller-executed steps in the student's rollouts
            o2, l2, r2, c2, e2 = collect_parallel(pool, spec, a.dagger_samples, (layers, obs_mean, obs_std, beta), 2000 + 100 * rnd, a.procs)
            log(f"round {rnd + 1}: student rollouts (beta {beta}) {len(o2)} samples, {r2} reaches, {c2} crashes in {e2} episodes")
            obs, lab = np.concatenate([obs, o2]), np.concatenate([lab, l2])
    log("saved", out)


if __name__ == "__main__":
    main()
