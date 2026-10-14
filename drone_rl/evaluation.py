"""The standard evaluation every run gets: python3 -m drone_rl eval <run>   (a few seconds per run).
A course is a list of waypoints flown in order under the strict reach rule (within 20 px, < 20 px/s, < 0.5 rad/s, held 25 steps); a waypoint not reached within
WAYPOINT_STEPS is skipped. There are fixed hand-designed courses (the same for every run, so runs compare directly) and random courses (fixed seeds, so also
comparable, but not tuned to). Metrics cover speed and success, flight style and control effort, and robustness (action noise, wild start states).
Everything is stored flat in the run's meta.json under 'metrics' (and eval.json), so any number can be a sortable column in the viewer."""
import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np

from drone_rl.config import TIME_STEP_DELTA, REACH_DIST, THRUSTER_NEUTRAL_ANGLE
from drone_rl.env import DroneEnv, make_spec, load_scheme

WAYPOINT_STEPS = 600  # 12 s per waypoint
MARGIN = 10.0
PROCS = 4
EVAL_VERSION = 2

# Gates (priority order: crash, tilt/rocking, outward nozzles) keep the earlier tier/score meaning.
CRASH_MAX, TILT_HOVER_MAX, TILT_STREAK_MAX, ROCK_MAX, OUTWARD_MIN = 0.02, 0.10, 0.10, 2.0, 0.70
TILT_DEG, SLOW_SPEED, STREAK_STEPS, SPIN_MIN, ROCK_GAP, OUT_DEG = 15.0, 30.0, 175, 0.3, 25, 10.0

# Fixed courses. ("frac", fx, fy): a fraction of the world (clamped MARGIN px inside the walls); ("rel", dx, dy): px from the previous waypoint (start = world center).
FIXED_COURSES = {
    "long_legs": [("frac", .9, .15), ("frac", .1, .85), ("frac", .9, .85), ("frac", .1, .15), ("frac", .5, .5)],
    "short_hops": [("rel", 150, 0), ("rel", 0, 150), ("rel", -150, 0), ("rel", 0, -150), ("rel", 100, 100), ("rel", -100, -100), ("rel", 200, 0), ("rel", -200, 0)],
    "wall_hugging": [("frac", .5, 0.0), ("frac", 1.0, .5), ("frac", .5, 1.0), ("frac", 0.0, .5), ("frac", 1.0, 0.0), ("frac", 0.0, 1.0)],
    "reversals": [("frac", .85, .5), ("frac", .15, .5), ("frac", .85, .5), ("frac", .15, .5), ("frac", .5, .1), ("frac", .5, .9), ("frac", .5, .1)],
    "close_precision": [("rel", 60, 0), ("rel", -60, 0), ("rel", 0, 60), ("rel", 0, -60), ("rel", 45, 45), ("rel", -45, -45), ("rel", 45, -45), ("rel", -45, 45)],
}
FIXED_REPS = 2  # each fixed course is flown with this many start-state seeds
RANDOM_SEEDS = (101, 102, 103, 104)  # random courses: 12 uniform waypoints each
RANDOM_WAYPOINTS = 12
ROBUST_CASES = {"noise02": {"noise": 0.02}, "noise05": {"noise": 0.05}, "wildstart": {"wild": True}}  # each flown on the first two random seeds


def course_points(name, world, seed):
    """Waypoints in px for a course name ('fixed:<course>' or 'random')."""
    W, H = world
    clamp = lambda p: (min(max(p[0], MARGIN), W - MARGIN), min(max(p[1], MARGIN), H - MARGIN))
    if name == "random":
        rng = random.Random(seed)
        return [(rng.uniform(MARGIN, W - MARGIN), rng.uniform(MARGIN, H - MARGIN)) for _ in range(RANDOM_WAYPOINTS)]
    pts, prev = [], (W / 2, H / 2)
    for kind, a, b in FIXED_COURSES[name.split(":", 1)[1]]:
        prev = clamp((a * W, b * H) if kind == "frac" else (prev[0] + a, prev[1] + b))
        pts.append(prev)
    return pts


def wild_start(env, rng):
    """A fast, tilted, spinning start anywhere in the middle of the world."""
    pos = np.array([rng.uniform(0.2, 0.8) * env.W, rng.uniform(0.2, 0.8) * env.H])
    vel = rng.uniform(-500, 500, 2).astype(np.float32)
    env.drone.shell.set_physical_state(pos, rng.uniform(-1.0, 1.0), vel, rng.uniform(-5, 5))


def run_episode(job):
    """One course flown once. Returns raw per-leg results and per-step sums (the aggregation happens in the parent)."""
    spec, policy_loader, course, seed, noise, wild = job
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    policy = policy_loader()
    module, scheme = load_scheme(spec["scheme"], spec["params"])
    env = DroneEnv(spec, module, scheme, max_steps=10 ** 9)
    env.set_ramp(1.0)
    env.set_eval_mode()
    env.reset()
    if wild:
        wild_start(env, rng)
    policy.reset(env)
    points = course_points(course[0], (env.W, env.H), course[1])
    sh, d = env.drone.shell, env.drone
    obs = env.observe()
    legs, crashed, crash_wall = [], False, None
    leg = {"steps": 0, "entered": None, "overshoot": 0.0}
    idx, steps, prev_act, prev_forces = 0, 0, None, list(d.thruster_forces)
    acc = {k: 0.0 for k in ("speed", "abs_tilt", "tilted", "slow", "slow_tilted", "abs_spin", "force_mean", "thrust", "force_jitter", "action_jitter", "nozzle_deg", "out")}
    peak_speed = peak_force = 0.0
    tilted_run, streak, spin_signs = 0, 0, []
    while idx < len(points):
        env.target = np.array(points[idx])
        act = np.asarray(policy.act(env, obs), dtype=np.float64)
        if noise:
            act = act + noise * rng.standard_normal(4)
        before = env.reaches
        obs, _, term, _ = env.step(act)
        steps += 1
        leg["steps"] += 1
        pos = np.array([float(sh.center[0]), float(sh.center[1])])
        dist = float(np.hypot(*(np.array(points[idx]) - pos)))
        if leg["entered"] is None and dist < REACH_DIST:
            leg["entered"] = leg["steps"]
        elif leg["entered"] is not None and dist > REACH_DIST:
            leg["overshoot"] = max(leg["overshoot"], dist)
        speed, angle, spin = sh.linear_velocity_magnitude, float(sh.angle), float(sh.angular_velocity)
        forces = list(d.thruster_forces)
        tilt = abs(math.degrees(angle) if abs(angle) <= math.pi else math.degrees((angle + math.pi) % (2 * math.pi) - math.pi))
        acc["speed"] += speed
        acc["abs_tilt"] += tilt
        is_tilted = tilt > TILT_DEG
        acc["tilted"] += is_tilted
        acc["slow"] += speed < SLOW_SPEED
        acc["slow_tilted"] += (speed < SLOW_SPEED) and is_tilted
        tilted_run = tilted_run + 1 if is_tilted else 0
        if tilted_run == STREAK_STEPS:
            streak += STREAK_STEPS
        elif tilted_run > STREAK_STEPS:
            streak += 1
        acc["abs_spin"] += abs(spin)
        acc["force_mean"] += max(forces)
        acc["thrust"] += sum(forces)
        acc["force_jitter"] += sum(abs(f - p) for f, p in zip(forces, prev_forces))
        acc["action_jitter"] += 0.0 if prev_act is None else float(np.abs(act - prev_act).mean())
        lo, ro = d.thruster_angles[0] - THRUSTER_NEUTRAL_ANGLE, THRUSTER_NEUTRAL_ANGLE - d.thruster_angles[1]
        acc["nozzle_deg"] += math.degrees((abs(lo) + abs(ro)) / 2)
        acc["out"] += (lo >= math.radians(OUT_DEG)) and (ro >= math.radians(OUT_DEG))
        peak_speed, peak_force = max(peak_speed, speed), max(peak_force, max(forces))
        spin_signs.append((steps, math.copysign(1, spin) if abs(spin) > SPIN_MIN else 0))
        prev_act, prev_forces = act, forces
        if term:
            crashed = True
            x, y = float(sh.center[0]), float(sh.center[1])
            crash_wall = "left" if x < 0 else "right" if x > env.W else "top" if y < 0 else "bottom"
            legs.append({"reached": False, "steps": leg["steps"], "overshoot": leg["overshoot"], "settle": None})
            break
        if env.reaches > before:
            legs.append({"reached": True, "steps": leg["steps"], "overshoot": leg["overshoot"], "settle": None if leg["entered"] is None else leg["steps"] - leg["entered"]})
            idx += 1
            leg = {"steps": 0, "entered": None, "overshoot": 0.0}
        elif leg["steps"] >= WAYPOINT_STEPS:
            legs.append({"reached": False, "steps": leg["steps"], "overshoot": leg["overshoot"], "settle": None})
            idx += 1
            leg = {"steps": 0, "entered": None, "overshoot": 0.0}
    signs = [(t, s) for t, s in spin_signs if s]
    rev = sum(1 for (t0, s0), (t1, s1) in zip(signs, signs[1:]) if s0 != s1 and t1 - t0 <= ROCK_GAP)
    return {"legs": legs, "steps": steps, "crashed": crashed, "crash_wall": crash_wall, "acc": acc, "peak_speed": peak_speed, "peak_force": peak_force,
            "streak": streak, "rev": rev, "n_waypoints": len(points)}


def _legs_stats(eps, prefix):
    legs = [leg for e in eps for leg in e["legs"]]
    reached = [leg for leg in legs if leg["reached"]]
    total_steps = sum(e["steps"] for e in eps)
    times = np.array([leg["steps"] * TIME_STEP_DELTA for leg in reached]) if reached else np.array([])
    n_points = sum(e["n_waypoints"] for e in eps)
    out = {f"{prefix}reached_frac": len(reached) / max(n_points, 1),
           f"{prefix}crash_rate": float(np.mean([e["crashed"] for e in eps])),
           f"{prefix}ttt_s_median": float(np.median(times)) if len(times) else None,
           f"{prefix}ttt_s_p90": float(np.percentile(times, 90)) if len(times) else None,
           f"{prefix}reaches_per_min": len(reached) / max(total_steps * TIME_STEP_DELTA / 60, 1e-9),
           f"{prefix}overshoot_px": float(np.mean([leg["overshoot"] for leg in reached])) if reached else None}
    settle = [leg["settle"] * TIME_STEP_DELTA for leg in reached if leg["settle"] is not None]
    out[f"{prefix}settle_s"] = float(np.mean(settle)) if settle else None
    return out


def _style_stats(eps):
    steps = sum(e["steps"] for e in eps)
    s = lambda key: sum(e["acc"][key] for e in eps)
    return {"mean_speed": s("speed") / steps, "peak_speed": max(e["peak_speed"] for e in eps), "mean_abs_tilt_deg": s("abs_tilt") / steps, "tilt_frac": s("tilted") / steps,
            "hover_tilt_frac": s("slow_tilted") / s("slow") if s("slow") else 0.0, "tilt_streak_frac": sum(e["streak"] for e in eps) / steps,
            "rock_rate": sum(e["rev"] for e in eps) / (steps * TIME_STEP_DELTA), "mean_abs_spin": s("abs_spin") / steps, "mean_force": s("force_mean") / steps,
            "peak_force": max(e["peak_force"] for e in eps), "mean_thrust": s("thrust") / steps, "force_jitter": s("force_jitter") / steps,
            "action_jitter": s("action_jitter") / steps, "nozzle_use_deg": s("nozzle_deg") / steps, "outward_frac": s("out") / steps}


def aggregate(results):
    """results: {(group, course, seed): episode dict} -> flat metrics dict."""
    m = {"eval_version": EVAL_VERSION}
    fixed = [r for (g, c, s), r in results.items() if g == "fixed"]
    clean = [r for (g, c, s), r in results.items() if g == "random"]
    m.update(_legs_stats(clean, ""))
    m.update(_legs_stats(fixed, "fixed_"))
    for course in FIXED_COURSES:
        eps = [r for (g, c, s), r in results.items() if g == "fixed" and c == course]
        st = _legs_stats(eps, "")
        m[f"fixed.{course}.reached_frac"], m[f"fixed.{course}.ttt_s"], m[f"fixed.{course}.crash_rate"] = st["reached_frac"], st["ttt_s_median"], st["crash_rate"]
    for case in ROBUST_CASES:
        eps = [r for (g, c, s), r in results.items() if g == case]
        st = _legs_stats(eps, "")
        m[f"robust.{case}.reached_frac"], m[f"robust.{case}.crash_rate"] = st["reached_frac"], st["crash_rate"]
    walls = [r["crash_wall"] for r in list(results.values()) if r["crash_wall"]]
    m["crash_walls"] = ",".join(sorted(set(walls))) if walls else ""
    m.update(_style_stats(clean))
    gates = [m["crash_rate"] <= CRASH_MAX, m["hover_tilt_frac"] <= TILT_HOVER_MAX and m["tilt_streak_frac"] <= TILT_STREAK_MAX and m["rock_rate"] <= ROCK_MAX,
             m["outward_frac"] >= OUTWARD_MIN]
    tier = 0
    for g in gates:
        if not g:
            break
        tier += 1
    m["tier"] = tier
    m["score"] = m["reaches_per_min"] if tier == 3 else m["reaches_per_min"] * 0.01 * (tier + 1)
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}


def build_jobs(spec, policy_loader):
    jobs, keys = [], []
    for name in FIXED_COURSES:
        for rep in range(FIXED_REPS):
            jobs.append((spec, policy_loader, ("fixed:" + name, 0), 10 + rep, 0.0, False))
            keys.append(("fixed", name, rep))
    for seed in RANDOM_SEEDS:
        jobs.append((spec, policy_loader, ("random", seed), seed, 0.0, False))
        keys.append(("random", "random", seed))
    for case, opts in ROBUST_CASES.items():
        for seed in RANDOM_SEEDS[:2]:
            jobs.append((spec, policy_loader, ("random", seed), seed + 500, opts.get("noise", 0.0), bool(opts.get("wild"))))
            keys.append((case, "random", seed))
    return jobs, keys


class _Loader:
    """Picklable callable that loads a run's policy inside the worker process."""

    def __init__(self, run_dir):
        self.run_dir = str(run_dir)

    def __call__(self):
        from drone_rl.policies import load_policy
        return load_policy(self.run_dir)


def evaluate_run(run_dir, procs=PROCS, write=True):
    """Runs the standard evaluation on a run folder; writes eval.json and returns the metrics dict."""
    t0 = time.time()
    run_dir = Path(run_dir)
    spec = make_spec(json.loads((run_dir / "spec.json").read_text()))
    jobs, keys = build_jobs(spec, _Loader(run_dir))
    with mp.get_context("spawn").Pool(procs) as pool:
        episodes = pool.map(run_episode, jobs, chunksize=1)
    metrics = aggregate(dict(zip(keys, episodes)))
    metrics["eval_seconds"] = round(time.time() - t0, 1)
    if write:
        (run_dir / "eval.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics
