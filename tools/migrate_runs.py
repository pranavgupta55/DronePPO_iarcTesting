"""One-off migration of the pre-refactor experiment runs (experiments/runs/<batch>/<name>, experiments/logs/<batch>/<name>.log) into the flat runs/<id>/ layout with a meta.json
each. Run once from the repo root: python3 tools/migrate_runs.py [--dry]. Timestamps come from the files' modification times (start = spec_in.json, finish = eval.json), so they are
the real training times. Groups, parents and tags are written here by hand from what each batch was for."""
import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))
from drone_rl import registry  # noqa: E402
from drone_rl.env import make_spec  # noqa: E402

OLD = REPO / "experiments"
WORLD_1200 = {"b000", "b001", "b002", "b003", "b004", "b005", "b006", "b007"}  # trained and evaluated in the 1200x800 world
CONTROL = "time_to_wall_barrier_x40_out30"
GROUPS = {  # batch -> (group, tags)
    "b000": ("calibration", ["calibration"]),
    "b001": ("scheme survey", ["survey", "default-weights"]),
    "b002": ("weight sweep", ["weight-sweep"]),
    "b003": ("outward reward and higher weights", ["weight-sweep", "outward-reward"]),
    "b004": ("replication of winners", ["replication"]),
    "b005": ("force rate tests", ["force-rate"]),
    "b006": ("physics and noise variant", ["physics-variant"]),
    "b007": ("approach and thrust reward overlays", ["reward-overlay"]),
    "b011": ("PPO fine-tune of imitation", ["ppo-finetune", "known-bug:normalizer-drift"]),
    "b012": ("PPO fine-tune of imitation", ["ppo-finetune", "diagnosis"]),
    "b013": ("PPO fine-tune with fixed normalizer", ["ppo-finetune"]),
    "b014": ("PPO fine-tune with fixed normalizer", ["ppo-finetune", "ablation"]),
    "b015": ("PPO fine-tune with fixed normalizer", ["ppo-finetune", "critic-warmup"]),
    "b016": ("nozzle use tests", ["ppo-finetune", "nozzles"]),
}


def parent_name(batch, name):
    """Name of the run this one branched from (by batch conventions), or None."""
    if batch == "b001":
        return "ts_default"
    if batch == "b002":
        base = name.rsplit("_x", 1)[0]
        return base
    if batch == "b003":
        for suffix in ("_out30", "_out3", "_x4more"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
    if batch == "b004":
        return name.rsplit("_s", 1)[0]
    if batch in ("b005", "b006", "b007"):
        return CONTROL
    if batch in ("b011", "b012", "b013"):
        return "bc_feat"
    if batch in ("b014",):
        return "t2_frozen_lr1e-05"
    if batch in ("b015", "b016"):
        return "t2_frozen_lr0.0001"
    return None


def when(path):
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    import csv
    batch_of = {r["name"]: r["batch"] for r in csv.DictReader(open(REPO / "docs" / "legacy_results.csv"))}
    batch_of.update({"bc_feat": "bc", "ttwb_force_rate_0.05_s2": "b005"})  # not in the old results table
    moves = []
    if (OLD / "runs").exists():  # phase 1: move the folders (skipped when a previous attempt already did it)
        for batch_dir in sorted((OLD / "runs").iterdir()):
            if batch_dir.name in ("failed_notes", "b008_failed_notes") or not batch_dir.is_dir():
                continue
            for run in sorted(batch_dir.iterdir()):
                if (run / "spec.json").exists():
                    moves.append((batch_dir.name, run))
    registry.RUNS.mkdir(exist_ok=True)
    used = set()
    for batch, run in moves:
        created = when(run / "spec_in.json") if (run / "spec_in.json").exists() else when(run / "spec.json")
        new_id, n = registry.run_id(run.name, created), 1
        while new_id in used or (registry.RUNS / new_id).exists():
            n += 1
            new_id = f"{registry.run_id(run.name, created)}-{n}"
        used.add(new_id)
        if not a.dry:
            shutil.move(str(run), str(registry.RUNS / new_id))
            log = OLD / "logs" / batch / f"{run.name}.log"
            if log.exists():
                shutil.move(str(log), str(registry.RUNS / new_id / "train.log"))
    # phase 2: describe every run folder that has no meta.json yet
    plan = []
    for dest in sorted(registry.RUNS.iterdir()):
        if not dest.is_dir() or (dest / "meta.json").exists() or not (dest / "spec.json").exists():
            continue
        name = json.loads((dest / "spec.json").read_text())["name"]
        batch = batch_of[name]
        created = when(dest / "spec_in.json") if (dest / "spec_in.json").exists() else when(dest / "spec.json")
        finished_src = [q for q in (dest / "eval.json", dest / "done.json", dest / "actor.pt") if q.exists()]
        plan.append((batch, Path(name), created, dest.name, when(finished_src[0]) if finished_src else created))
    print(len(plan), "runs to describe")
    if a.dry:
        for b, r, cr, c, f in plan[:5]:
            print(b, r.name, "->", c, cr.isoformat(), f.isoformat())
        return
    ids = {(batch, run.name): new_id for batch, run, created, new_id, finished in plan}
    # the hand-written controller as its own run, the root of the imitation family
    ctrl = registry.RUNS / registry.run_id("bang-bang-controller", datetime.fromisoformat(subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%aI", "--", "experiments/bangbang_controller.py"], capture_output=True, text=True, cwd=REPO).stdout.strip().splitlines()[-1]))
    ctrl.mkdir()
    blurb = "Bang-bang controller\nHand-written, not trained. Turns to face the target, thrusts hard, flips and thrusts back to brake. The teacher for the imitation runs."
    ctrl_spec = make_spec({"name": "bang-bang-controller", "scheme": "time_shaping", "kind": "controller", "world": [2400, 1600], "force_rate": 0.5, "blurb": blurb,
                           "group": "bang-bang imitation", "tags": ["controller", "hand-written"], "mirror": False})
    (ctrl / "spec.json").write_text(json.dumps(ctrl_spec, indent=1))
    (ctrl / "spec_in.json").write_text(json.dumps(ctrl_spec))
    ctrl_meta = {"id": ctrl.name, "name": "bang-bang-controller", "title": "Bang-bang controller", "blurb_text": blurb.split("\n", 1)[1], "group": "bang-bang imitation", "parent": None,
                 "tags": ["controller", "hand-written"], "kind": "controller", "status": "ok", "created": datetime.fromisoformat(ctrl.name[:10] + "T" + ctrl.name[11:13] + ":" + ctrl.name[13:15] + ":00").astimezone().isoformat(),
                 "finished": None, "world": [2400, 1600], "model": {"obs_dim": 24, "hidden": [], "params": 0}, "spec": ctrl_spec, "legacy": {"created_source": "date of the first commit containing the controller"}, "training": {}, "metrics": {}, "flyable": True}
    registry.write_meta(ctrl, ctrl_meta)
    bc_id = ids.get(("bc", "bc_feat"))
    for batch, run, created, new_id, finished in plan:
        dest = registry.RUNS / new_id
        spec = json.loads((dest / "spec.json").read_text())
        old_eval = json.loads((dest / "eval.json").read_text()) if (dest / "eval.json").exists() else {}
        blurb_file = dest / "blurb.txt"
        blurb = blurb_file.read_text().strip() if blurb_file.exists() else f"{run.name}\n(no description recorded)"
        title, text = registry.split_blurb(blurb)
        group, tags = GROUPS.get(batch, ("imitation of the bang-bang controller", ["imitation", "dagger"]))
        kind = "imitation" if batch == "bc" else ("ppo-finetune" if spec.get("init_from") else "ppo")
        parent = None
        pn = parent_name(batch, run.name)
        if batch == "bc":
            parent = ctrl.name
        elif pn:
            cands = [i for (b, n), i in ids.items() if n == pn]
            parent = cands[0] if cands else None
        world = [1200, 800] if batch in WORLD_1200 else [2400, 1600]
        if spec.get("init_from"):
            spec["init_from"] = bc_id
        spec.update({"world": world, "group": group, "tags": tags, "parent": parent, "kind": kind, "blurb": blurb})
        full = make_spec(spec)
        (dest / "spec.json").write_text(json.dumps(full, indent=1))
        from drone_rl.config import OBS_BASE_DIM
        from drone_rl.env import load_scheme
        module, _ = load_scheme(full["scheme"], full["params"])
        obs_dim = OBS_BASE_DIM + int(module.EXTRA_OBS)
        has_model = (dest / "actor.pt").exists()
        status = "ok" if has_model and old_eval else "failed"
        done = json.loads((dest / "done.json").read_text()) if (dest / "done.json").exists() else {}
        meta = {"id": new_id, "name": run.name, "title": title, "blurb_text": text, "group": group, "parent": parent, "tags": tags, "kind": kind, "status": status,
                "created": created.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"), "world": world,
                "model": {"obs_dim": obs_dim, "hidden": list(full["hidden"]), "params": registry.number_params(full["hidden"], obs_dim)},
                "spec": {k: v for k, v in full.items() if k != "blurb"}, "legacy": {"batch": batch, "original_path": f"experiments/runs/{batch}/{run.name}", "eval": old_eval},
                "training": {"epochs": done.get("epochs"), "wall_seconds": done.get("wall_seconds"), "samples_per_epoch": full["steps_per_epoch"] * full["num_envs"]}, "metrics": {}, "flyable": has_model}
        if (dest / "epochs.csv").exists():
            from drone_rl.runner import training_summary
            meta["training"]["summary"] = training_summary(dest / "epochs.csv")
        registry.write_meta(dest, meta)
    for notes in ("failed_notes", "b008_failed_notes"):
        if (OLD / "runs" / notes).exists():
            shutil.move(str(OLD / "runs" / notes), str(REPO / "archive" / notes))
    registry.build_index()
    print("migrated", len(plan), "runs + the controller; index rebuilt")


if __name__ == "__main__":
    main()
