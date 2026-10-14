"""One entry point for everything. `python3 -m drone_rl --help` lists the commands."""
import importlib
import sys

COMMANDS = {
    "batch": ("drone_rl.batch", "run a specs file: train + evaluate every run in it"),
    "train-one": ("drone_rl.train_one", "train a single spec (used by batch): train-one <spec.json> <run_dir>"),
    "eval": ("drone_rl.eval_cli", "evaluate runs on the standard courses: eval <run> [<run> ...] | eval --all"),
    "imitate": ("drone_rl.imitation", "behavior-clone the bang-bang controller into a network (DAgger)"),
    "baseline": ("drone_rl.baseline", "evaluate the hand-written controller and compare stored baselines"),
    "check-scheme": ("drone_rl.check_scheme", "validate a reward scheme: check-scheme <name> [params_json]"),
    "index": ("drone_rl.index_cli", "rebuild runs/index.json from the runs' meta.json files"),
    "report": ("drone_rl.report", "summary table and charts of the run index"),
    "bench": ("drone_rl.bench", "throughput benchmark for parallel training sessions"),
    "view": ("drone_rl.viewer.app", "open the viewer (arrow keys switch runs, Tab opens the run menu)"),
}


def main(argv):
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: python3 -m drone_rl <command> [args]\n")
        for name, (_, text) in COMMANDS.items():
            print(f"  {name:13s} {text}")
        sys.exit(0 if not argv or argv[0] in ("-h", "--help") else 2)
    module = importlib.import_module(COMMANDS[argv[0]][0])
    sys.argv = [f"drone_rl {argv[0]}"] + argv[1:]
    module.main()
