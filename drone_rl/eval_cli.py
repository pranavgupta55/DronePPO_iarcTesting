"""python3 -m drone_rl eval <run> [<run> ...] [--all] [--procs N]: standard evaluation, written to each run's eval.json and meta.json."""
import argparse

from drone_rl import registry, runner
from drone_rl.evaluation import evaluate_run, PROCS


def main():
    ap = argparse.ArgumentParser(prog="drone_rl eval")
    ap.add_argument("runs", nargs="*", help="run ids or names")
    ap.add_argument("--all", action="store_true", help="evaluate every flyable run again")
    ap.add_argument("--missing", action="store_true", help="only runs without current-version metrics")
    ap.add_argument("--procs", type=int, default=PROCS)
    a = ap.parse_args()
    from pathlib import Path
    dirs = [d for d, m in registry.iter_runs() if m.get("flyable", True) and m.get("status") == "ok"] if a.all or a.missing else [registry.find_run(r) if not Path(r).is_dir() else Path(r) for r in a.runs]
    from drone_rl.evaluation import EVAL_VERSION
    for d in dirs:
        meta = registry.read_meta(d)
        if a.missing and meta.get("metrics", {}).get("eval_version") == EVAL_VERSION:
            continue
        m = evaluate_run(d, a.procs)
        if (d / "meta.json").exists():
            runner.finalize_run(d, meta.get("status", "ok"))
        print(f"{d.name}: {m['reaches_per_min']:.1f} reaches/min, crash {m['crash_rate']:.2f}, median TtT {m['ttt_s_median']}, tier {m['tier']} ({m['eval_seconds']}s)", flush=True)
    registry.build_index()


if __name__ == "__main__":
    main()
