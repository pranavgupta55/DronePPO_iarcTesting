"""Throughput benchmark: python3 bench.py  -> reports/bench.md
Runs 15-epoch time_shaping trainings exactly as run_batch would, for several (parallel sessions, envs per session) combos."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
EPOCHS = 15
CONFIGS = [(1, 8), (2, 4), (3, 3), (5, 2)]  # (P concurrent sessions, num_envs each); all keep 32768 samples per epoch


def main():
    root = HERE / "reports" / "bench_work"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    rows = []
    for P, envs in CONFIGS:
        name = f"bench_P{P}"
        specs = [{"name": f"{name}_{i}", "scheme": "time_shaping", "epochs": EPOCHS, "num_envs": envs, "seed": i} for i in range(P)]
        (root / f"{name}.json").write_text(json.dumps({"batch": name, "specs": specs}))
        t0 = time.time()
        subprocess.run([sys.executable, str(HERE / "run_batch.py"), str(root / f"{name}.json"), "--parallel", str(P), "--results", str(root / "results.csv"),
                        "--runs-root", str(root / "runs")], check=True, cwd=HERE, stdout=subprocess.DEVNULL)
        wall = time.time() - t0
        spe, startup, evals = [], [], []
        for s in specs:
            d = root / "runs" / name / s["name"]
            lines = (d / "epochs.csv").read_text().splitlines()
            times = [float(l.split(",")[1]) for l in lines[1:]]  # cumulative seconds per epoch
            per = (times[-1] - times[4]) / (len(times) - 5)  # epochs 6..15, excludes startup
            spe.append(per)
            startup.append(times[0] - per)
            evals.append(json.loads((d / "eval.json").read_text())["eval_seconds"])
        per, st, ev = sum(spe) / P, sum(startup) / P, sum(evals) / P
        run_s = st + 300 * per + ev
        rows.append((P, envs, per, st, ev, run_s / 60, P * 3600 / run_s, wall))
        print(f"P={P} envs={envs}: {per:.2f}s/epoch, est {run_s / 60:.1f} min/run, {P * 3600 / run_s:.0f} runs/h", flush=True)
    t0 = time.time()
    sys.path.insert(0, str(HERE))
    from evaluate import evaluate
    evaluate(root / "runs" / "bench_P1" / "bench_P1_0")
    solo_eval = time.time() - t0
    best = max(rows, key=lambda r: r[6])
    md = ["# Throughput benchmark", "", f"{EPOCHS}-epoch time_shaping trainings run by run_batch.py with P concurrent sessions; 32768 samples/epoch in all cases.",
          "Per-session numbers are averages. est run = startup + 300 x s/epoch + eval (eval measured under load).", "",
          "| P | envs/session | s/epoch | startup s | eval s | est min/run | runs/hour | bench wall s |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {P} | {e} | {p:.2f} | {s:.1f} | {v:.1f} | {m:.1f} | {r:.0f} | {w:.0f} |" for P, e, p, s, v, m, r, w in rows]
    md += ["", f"Best: P={best[0]}, workers (num_envs)={best[1]} -> {best[6]:.0f} runs/hour.", f"Standalone evaluate(): {solo_eval:.1f} s."]
    (HERE / "reports" / "bench.md").write_text("\n".join(md) + "\n")
    shutil.rmtree(root, ignore_errors=True)
    print("\n".join(md))


if __name__ == "__main__":
    main()
