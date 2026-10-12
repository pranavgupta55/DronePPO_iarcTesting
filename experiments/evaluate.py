"""Fixed evaluation of a trained run: python3 evaluate.py <run_dir>  (or evaluate(run_dir) -> dict; also writes <run_dir>/eval.json).
Protocol: 30 episodes x 2048 steps, fixed seeds, mean action, no mirroring, strict reach rule, uniform targets (margin 10)."""
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

from core.config import TIME_STEP_DELTA
from core.env import DroneEnv, NumpyPolicy, make_spec, load_scheme, NORM_CLIP

# Gate thresholds (adjust as you like). Gates are checked in priority order: crash, tilt, outward.
CRASH_MAX = 0.02  # share of episodes that end by leaving the world
TILT_HOVER_MAX = 0.10  # share of slow steps (< 30 px/s) tilted more than 15 deg
TILT_STREAK_MAX = 0.10  # share of all steps in tilt streaks (>= 3.5 s with |angle| > 15 deg)
ROCK_MAX = 2.0  # spin-direction reversals per second (a steady 1 Hz rocking gives 2)
OUTWARD_MIN = 0.70  # share of steps with both nozzles at least 10 deg outward

N_EPISODES, EP_STEPS, SEED0, PROCS = 30, 2048, 1000, 4
TILT_DEG, SLOW_SPEED, STREAK_STEPS, SPIN_MIN, ROCK_GAP, OUT_DEG = 15.0, 30.0, 175, 0.3, 25, 10.0


def run_episode(args):
    """One evaluation episode; returns raw per-episode counts."""
    spec, layers, obs_mean, obs_std, seed = args
    random.seed(seed)
    np.random.seed(seed)
    module, scheme = load_scheme(spec["scheme"], spec["params"])
    env = DroneEnv(spec, module, scheme, max_steps=EP_STEPS)
    env.set_ramp(1.0)
    env.set_eval_mode()
    policy = NumpyPolicy(layers, np.zeros(4))
    obs = env.reset()
    n = 0
    speed, angle, spin, lo, ro = (np.zeros(EP_STEPS) for _ in range(5))
    reach_steps, crashed = [], False
    for t in range(EP_STEPS):
        reaches = env.reaches
        obs, _, term, trunc = env.step(policy.mean(np.clip((obs - obs_mean) / obs_std, -NORM_CLIP, NORM_CLIP)))
        sh = env.drone.shell
        speed[t], angle[t], spin[t] = sh.linear_velocity_magnitude, sh.angle, sh.angular_velocity
        lo[t] = env.drone.thruster_angles[0] - math.pi / 2
        ro[t] = math.pi / 2 - env.drone.thruster_angles[1]
        if env.reaches > reaches:
            reach_steps.append(t + 1)
        n = t + 1
        if term:
            crashed = True
        if term or trunc:
            break
    speed, angle, spin, lo, ro = speed[:n], angle[:n], spin[:n], lo[:n], ro[:n]
    tilted = np.abs(angle) > math.radians(TILT_DEG)
    slow = speed < SLOW_SPEED
    # tilt streaks: runs of consecutive tilted steps of at least STREAK_STEPS
    padded = np.concatenate([[0], tilted.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    streak = int(sum(e - s for s, e in zip(edges[::2], edges[1::2]) if e - s >= STREAK_STEPS))
    # rocking: sign reversals of the spin between consecutive fast-spinning steps no more than ROCK_GAP steps apart
    idx = np.flatnonzero(np.abs(spin) > SPIN_MIN)
    rev = int(np.sum((np.sign(spin[idx[1:]]) != np.sign(spin[idx[:-1]])) & (np.diff(idx) <= ROCK_GAP))) if len(idx) > 1 else 0
    out = (lo >= math.radians(OUT_DEG)) & (ro >= math.radians(OUT_DEG))
    gaps = np.diff([0] + reach_steps)
    return {"steps": n, "reaches": len(reach_steps), "reach_gap_sum": int(gaps.sum()) if len(gaps) else 0, "crashed": crashed,
            "speed": float(speed.sum()), "abs_tilt": float(np.abs(angle).sum()), "slow": int(slow.sum()), "slow_tilted": int((slow & tilted).sum()),
            "streak": streak, "rev": rev, "abs_spin": float(np.abs(spin).sum()), "out": int(out.sum()), "out_deg": float(np.degrees((lo + ro) / 2).sum())}


def load_policy(run_dir):
    import torch  # only needed here, so the worker processes never load it
    state = torch.load(run_dir / "actor.pt", map_location="cpu")
    keys = sorted({int(k.split(".")[1]) for k in state if k.startswith("net.")})
    layers = [(state[f"net.{i}.weight"].numpy().astype(np.float64), state[f"net.{i}.bias"].numpy().astype(np.float64)) for i in keys]
    norms = np.load(run_dir / "norm.npz")
    return layers, norms["obs_mean"], np.sqrt(norms["obs_var"] + 1e-8)


def evaluate(run_dir):
    t0 = time.time()
    run_dir = Path(run_dir)
    spec = make_spec(json.loads((run_dir / "spec.json").read_text()))
    layers, obs_mean, obs_std = load_policy(run_dir)
    if not spec["obs_norm"]:
        obs_mean, obs_std = np.zeros_like(obs_mean), np.ones_like(obs_std)
    jobs = [(spec, layers, obs_mean, obs_std, SEED0 + i) for i in range(N_EPISODES)]
    with mp.get_context("spawn").Pool(PROCS) as pool:
        eps = pool.map(run_episode, jobs, chunksize=1)
    total = sum(e["steps"] for e in eps)
    reaches = sum(e["reaches"] for e in eps)
    s = lambda key: sum(e[key] for e in eps)
    m = {"reaches_per_ep": reaches / N_EPISODES,
         "steps_per_reach": s("reach_gap_sum") / reaches if reaches else float(total),  # mean gap between reaches, counting the episode start
         "crash_rate": s("crashed") / N_EPISODES,
         "mean_speed": s("speed") / total, "mean_abs_tilt_deg": math.degrees(s("abs_tilt") / total),
         "hover_tilt_frac": s("slow_tilted") / s("slow") if s("slow") else 0.0, "tilt_streak_frac": s("streak") / total,
         "rock_rate": s("rev") / (total * TIME_STEP_DELTA), "mean_abs_ang_vel": s("abs_spin") / total,
         "outward_frac": s("out") / total, "mean_outward_deg": s("out_deg") / total}
    gates = [m["crash_rate"] <= CRASH_MAX,
             m["hover_tilt_frac"] <= TILT_HOVER_MAX and m["tilt_streak_frac"] <= TILT_STREAK_MAX and m["rock_rate"] <= ROCK_MAX,
             m["outward_frac"] >= OUTWARD_MIN]
    tier = 0
    for g in gates:
        if not g:
            break
        tier += 1
    m["tier"] = tier
    m["score"] = m["reaches_per_ep"] if tier == 3 else m["reaches_per_ep"] * 0.01 * (tier + 1)
    m["eval_seconds"] = round(time.time() - t0, 1)
    (run_dir / "eval.json").write_text(json.dumps(m, indent=1))
    return m


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1])))
