"""Runs a batch of training+evaluation sessions.
python3 run_batch.py <batch.json> [--parallel P] [--workers N] [--batch NAME] [--results CSV] [--runs-root DIR]
batch.json = {"batch": "b001", "specs": [spec, ...]} or a plain list of specs (then give --batch). Resumable: specs whose run already has eval.json are skipped."""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
TRAIN_TIMEOUT, EVAL_TIMEOUT = 25 * 60, 5 * 60
sys.path.insert(0, str(HERE))
from core.env import DEFAULTS, NON_HYPER_KEYS, make_spec  # noqa: E402 (torch-free)

HYPER_KEYS = [k for k in DEFAULTS if k not in NON_HYPER_KEYS]
EVAL_KEYS = ["reaches_per_ep", "steps_per_reach", "crash_rate", "mean_speed", "mean_abs_tilt_deg", "hover_tilt_frac", "tilt_streak_frac", "rock_rate",
             "mean_abs_ang_vel", "outward_frac", "mean_outward_deg", "tier", "score"]
TRAIN_KEYS = ["final_return", "final_length", "final_reaches_per_epoch"]
COLUMNS = ["batch", "name", "status", "seed", "scheme", "params"] + HYPER_KEYS + EVAL_KEYS + TRAIN_KEYS + ["wall_seconds", "error"]
csv_lock = threading.Lock()


def run_subprocess(cmd, log, timeout):
    """Runs cmd with output appended to log; kills the whole process group (incl. env workers) on timeout. Returns (ok, reason)."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE, env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            f.write(f"\nTIMEOUT after {timeout}s\n")
            return False, "timeout"
        if code != 0:
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # make sure no worker is left behind
            except ProcessLookupError:
                pass
            return False, f"exit code {code}"
    return True, ""


def tail(log, n=6):
    try:
        return " | ".join(l.strip() for l in Path(log).read_text().strip().splitlines()[-n:])[-600:]
    except OSError:
        return ""


def append_row(results, row):
    with csv_lock:
        new = not results.exists() or results.stat().st_size == 0
        with open(results, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def run_one(batch, spec, runs_root, logs_root, results):
    t0 = time.time()
    name = spec["name"]
    run_dir, log = runs_root / name, logs_root / f"{name}.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {"batch": batch, "name": name, "seed": spec["seed"], "scheme": spec["scheme"], "params": json.dumps(spec["params"], separators=(",", ":"), sort_keys=True),
           **{k: json.dumps(spec[k], separators=(",", ":")) if isinstance(spec[k], (list, dict)) else spec[k] for k in HYPER_KEYS}}
    spec_file = run_dir / "spec_in.json"
    spec_file.write_text(json.dumps(spec))
    ok, why = run_subprocess([sys.executable, "train_one.py", str(spec_file), str(run_dir)], log, TRAIN_TIMEOUT)
    status = "ok"
    if not ok:
        status = "timeout" if why == "timeout" else "train_failed"
    else:
        ok, why = run_subprocess([sys.executable, "evaluate.py", str(run_dir)], log, EVAL_TIMEOUT)
        if not ok:
            status = "eval_failed"
    if status == "ok":
        row.update(json.loads((run_dir / "eval.json").read_text()))
        done = json.loads((run_dir / "done.json").read_text())
        row.update({k: done.get(k) for k in TRAIN_KEYS})
    else:
        row["error"] = f"{why}: {tail(log)}"
    row["status"], row["wall_seconds"] = status, round(time.time() - t0, 1)
    append_row(results, row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_file")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--workers", type=int, default=None, help="num_envs for specs that don't set it")
    ap.add_argument("--batch", default=None)
    ap.add_argument("--results", default=str(HERE / "results.csv"))
    ap.add_argument("--runs-root", default=str(HERE / "runs"))
    args = ap.parse_args()

    data = json.loads(Path(args.batch_file).read_text())
    batch = args.batch or (data.get("batch") if isinstance(data, dict) else None)
    specs = data["specs"] if isinstance(data, dict) else data
    if not batch:
        sys.exit("batch name missing: put it in the json or pass --batch")
    runs_root, logs_root, results = Path(args.runs_root) / batch, HERE / "logs" / batch, Path(args.results)
    logs_root.mkdir(parents=True, exist_ok=True)
    todo, names = [], set()
    for i, s in enumerate(specs):
        s = dict(s)
        s.setdefault("name", f"{batch}_{i:03d}")
        if args.workers and "num_envs" not in s:
            s["num_envs"] = args.workers
        if s["name"] in names:
            sys.exit(f"duplicate run name {s['name']}")
        names.add(s["name"])
        try:
            full = make_spec(s)
        except Exception as e:  # a bad spec is recorded as a failure, not a crash
            append_row(results, {"batch": batch, "name": s["name"], "status": "bad_spec", "scheme": s.get("scheme"), "seed": s.get("seed"), "error": repr(e)})
            print(f"[{batch}] {s['name']}: bad_spec {e!r}", flush=True)
            continue
        if (runs_root / full["name"] / "eval.json").exists():
            continue
        todo.append(full)
    print(f"[{batch}] {len(todo)} to run ({len(specs) - len(todo)} skipped), parallel {args.parallel}", flush=True)

    t0, counts = time.time(), {}
    with ThreadPoolExecutor(args.parallel) as pool:
        futures = [pool.submit(run_one, batch, s, runs_root, logs_root, results) for s in todo]
        for fut in futures:
            try:
                r = fut.result()
            except Exception as e:  # never stop the batch
                print(f"[{batch}] runner error {e!r}", flush=True)
                continue
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] == "ok":
                print(f"[{batch}] {r['name']}: reaches {r['reaches_per_ep']:.2f} tier {r['tier']} score {r['score']:.2f} crash {r['crash_rate']:.2f} {r['wall_seconds']:.0f}s", flush=True)
            else:
                print(f"[{batch}] {r['name']}: {r['status']} {r.get('error', '')[-140:]}", flush=True)
    print(f"[{batch}] finished in {(time.time() - t0) / 60:.1f} min: {counts}", flush=True)


if __name__ == "__main__":
    main()
