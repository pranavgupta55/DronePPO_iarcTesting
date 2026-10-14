"""Runs a batch of training + evaluation sessions: python3 -m drone_rl batch specs/<file>.json [--parallel P] [--workers N]
A specs file is {"batch": "b017", "group": "...", "tags": [...], "parent": "<run name>", "specs": [spec, ...]}; group / tags / parent at the top are defaults for every spec.
Every spec needs a name, a blurb ("short name\\nwhat it tests, what changed from the branch point"), a group and at least one tag (see registry.check_run_spec).
Each run lands in runs/<date>_<name>/ with its meta.json. Resumable: a spec whose run (same batch and name) already finished ok is skipped."""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from drone_rl import registry, runner

TRAIN_TIMEOUT, EVAL_TIMEOUT = 25 * 60, 5 * 60
REPO = registry.REPO
print_lock = threading.Lock()


def run_subprocess(cmd, log, timeout):
    """Runs cmd with output appended to log; kills the whole process group (incl. env workers) on timeout. Returns (ok, reason)."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=env, start_new_session=True)
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


def run_one(batch, spec, root):
    t0 = time.time()
    run_dir, full = runner.prepare_run(spec, batch=batch, root=root)
    log = run_dir / "train.log"
    spec_file = run_dir / "spec_in.json"
    ok, why = run_subprocess([sys.executable, "-m", "drone_rl", "train-one", str(spec_file), str(run_dir)], log, TRAIN_TIMEOUT)
    status = "ok"
    if not ok:
        status = "timeout" if why == "timeout" else "train_failed"
    else:
        ok, why = run_subprocess([sys.executable, "-m", "drone_rl", "eval", str(run_dir)], log, EVAL_TIMEOUT)
        if not ok:
            status = "eval_failed"
    meta = runner.finalize_run(run_dir, status, None if status == "ok" else f"{why}: {tail(log)}")
    meta["wall"] = round(time.time() - t0, 1)
    return run_dir, meta


def load_specs(path, batch_override=None):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        data = {"specs": data}
    batch = batch_override or data.get("batch") or Path(path).stem
    defaults = {k: data[k] for k in ("group", "tags", "parent") if k in data}
    specs = []
    for i, s in enumerate(data["specs"]):
        s = {**defaults, **s}
        s.setdefault("name", f"{batch}_{i:03d}")
        specs.append(s)
    return batch, specs


def already_done(batch, name, root):
    return any(m.get("name") == name and m.get("legacy", {}).get("batch") == batch and m.get("status") == "ok" for _, m in registry.iter_runs(root))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drone_rl batch")
    ap.add_argument("specs_file")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--workers", type=int, default=None, help="num_envs for specs that don't set it")
    ap.add_argument("--batch", default=None)
    ap.add_argument("--root", default=str(registry.RUNS))
    args = ap.parse_args(argv)
    batch, specs = load_specs(args.specs_file, args.batch)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    todo, names = [], set()
    for s in specs:
        if args.workers and "num_envs" not in s:
            s["num_envs"] = args.workers
        if s["name"] in names:
            sys.exit(f"duplicate run name {s['name']}")
        names.add(s["name"])
        registry.check_run_spec(s)  # fail before anything starts
        if not already_done(batch, s["name"], root):
            todo.append(s)
    print(f"[{batch}] {len(todo)} to run ({len(specs) - len(todo)} skipped), parallel {args.parallel}", flush=True)
    t0, counts = time.time(), {}
    with ThreadPoolExecutor(args.parallel) as pool:
        futures = [pool.submit(run_one, batch, s, root) for s in todo]
        for fut in futures:
            try:
                run_dir, meta = fut.result()
            except Exception as e:  # never stop the batch
                print(f"[{batch}] runner error {e!r}", flush=True)
                continue
            counts[meta["status"]] = counts.get(meta["status"], 0) + 1
            m = meta.get("metrics", {})
            if meta["status"] == "ok":
                print(f"[{batch}] {meta['name']}: {m.get('reaches_per_min', 0):.1f} reaches/min, crash {m.get('crash_rate', 0):.2f}, tier {m.get('tier')} ({meta['wall']:.0f}s)", flush=True)
            else:
                print(f"[{batch}] {meta['name']}: {meta['status']} {meta.get('error', '')[-140:]}", flush=True)
    registry.build_index(root)
    print(f"[{batch}] finished in {(time.time() - t0) / 60:.1f} min: {counts}", flush=True)


if __name__ == "__main__":
    main()
