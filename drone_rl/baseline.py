"""Regression check on the physics and environment: python3 -m drone_rl baseline [--update]
Re-evaluates the baseline runs (the hand-written controller and the first imitation network) on the standard courses and compares a few headline numbers with the
ones stored in docs/baselines.json. A drift beyond the tolerance means something in the physics, the environment or the evaluation changed. --update stores new numbers."""
import argparse
import json

from drone_rl import registry
from drone_rl.evaluation import evaluate_run

BASELINES = registry.REPO / "docs" / "baselines.json"
RUNS = ["bang-bang-controller", "bc_feat"]
KEYS = {"reaches_per_min": 0.20, "crash_rate": None, "ttt_s_median": 0.20, "mean_speed": 0.20}  # relative tolerance; None = absolute 0.10
ABS_CRASH = 0.10


def main():
    ap = argparse.ArgumentParser(prog="drone_rl baseline")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    stored = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}
    current, bad = {}, 0
    for name in RUNS:
        m = evaluate_run(registry.find_run(name), write=False)
        current[name] = {k: m[k] for k in KEYS}
        if a.update or name not in stored:
            continue
        for key, tol in KEYS.items():
            old, new = stored[name][key], m[key]
            drift = abs(new - old) > ABS_CRASH if tol is None else abs(new - old) > tol * max(abs(old), 1e-9)
            bad += drift
            print(f"{'DRIFT' if drift else 'ok   '} {name:22s} {key:16s} stored {old:8.3f}  now {new:8.3f}")
    if a.update or not stored:
        BASELINES.write_text(json.dumps(current, indent=1) + "\n")
        print("stored baselines in", BASELINES)
    elif bad:
        raise SystemExit(f"{bad} baseline number(s) drifted")


if __name__ == "__main__":
    main()
