"""Creating a run folder before training and finishing its meta.json afterwards (training summary, model size, evaluation metrics)."""
import json
from pathlib import Path

import numpy as np

from drone_rl import registry
from drone_rl.config import OBS_BASE_DIM
from drone_rl.env import make_spec, load_scheme

CURVE_POINTS = (0.25, 0.5, 0.75, 1.0)


def training_summary(epochs_csv):
    """Numbers that describe the training curve (reaches per epoch is the learning signal): value at 25/50/75/100% of training, best epoch, final spread, biggest drop,
    epochs needed to reach half of the final level, plus the last losses and learning rate."""
    import csv
    rows = list(csv.DictReader(open(epochs_csv)))
    if not rows:
        return {}
    num = lambda key: np.array([float(r[key]) if r.get(key) not in (None, "", "nan") else np.nan for r in rows])
    reaches, ret, length = num("total_reaches"), num("mean_return"), num("mean_length")
    n = len(rows)
    smooth = np.convolve(reaches, np.ones(5) / 5, mode="valid") if n >= 5 else reaches
    final = reaches[-10:]
    half = np.flatnonzero(smooth >= 0.5 * np.nanmean(final))
    summary = {"epochs_logged": n, "final_reaches_mean": float(np.nanmean(final)), "final_reaches_std": float(np.nanstd(final)), "best_epoch": int(np.nanargmax(smooth)) + 1,
               "best_reaches": float(np.nanmax(smooth)), "epochs_to_half_final": int(half[0]) + 1 if len(half) else None,
               "biggest_drop": float(np.nanmax(np.maximum.accumulate(smooth) - smooth)) if len(smooth) else 0.0,
               "final_return": float(np.nanmean(ret[-10:])), "final_episode_length": float(np.nanmean(length[-10:])),
               "episodes_per_epoch": float(np.nanmean(num("episodes"))), "last_pi_loss": float(num("pi_loss")[-1]), "last_v_loss": float(num("v_loss")[-1]),
               "last_actor_lr": float(num("actor_lr")[-1])}
    for q in CURVE_POINTS:
        summary[f"reaches_at_{int(q * 100)}pct"] = float(np.nanmean(reaches[max(0, int(q * n) - 5):max(1, int(q * n))]))
    return summary


def prepare_run(spec, batch=None, root=None, when=None):
    """Creates runs/<id>/ for a new run and writes its first meta.json (status 'running'). Returns (run_dir, full spec)."""
    registry.check_run_spec(spec)
    spec = dict(spec)
    parent = spec.get("parent")
    if parent:
        try:
            spec["parent"] = registry.find_run(parent, root).name
        except KeyError:
            pass
    full = make_spec(spec)
    run_dir = registry.new_run_dir(full["name"], when, root)
    title, text = registry.split_blurb(full["blurb"])
    module, _ = load_scheme(full["scheme"], full["params"])
    obs_dim = OBS_BASE_DIM + int(module.EXTRA_OBS)
    meta = {"id": run_dir.name, "name": full["name"], "title": title, "blurb_text": text, "group": full["group"], "parent": full["parent"], "tags": list(full["tags"]),
            "kind": registry.kind_of(full), "status": "running", "created": registry.now_iso(), "finished": None, "world": list(full["world"]),
            "model": {"obs_dim": obs_dim, "hidden": list(full["hidden"]), "params": registry.number_params(full["hidden"], obs_dim)},
            "spec": {k: v for k, v in full.items() if k not in ("blurb",)}, "legacy": {"batch": batch} if batch else {}, "training": {}, "metrics": {}}
    (run_dir / "spec_in.json").write_text(json.dumps(full))
    registry.write_meta(run_dir, meta)
    return run_dir, full


def finalize_run(run_dir, status="ok", error=None):
    """Fills meta.json from the files the training and the evaluation left in the run folder."""
    run_dir = Path(run_dir)
    changes = {"status": status}
    if registry.read_meta(run_dir).get("finished") is None and registry.read_meta(run_dir).get("kind") != "controller":
        changes["finished"] = registry.now_iso()  # set once, when the run first finishes (re-evaluating later must not move it)
    if error:
        changes["error"] = error[-600:]
    done = run_dir / "done.json"
    if done.exists():
        d = json.loads(done.read_text())
        changes["training"] = {"epochs": d.get("epochs"), "wall_seconds": d.get("wall_seconds"), "samples_per_epoch": None}
    if (run_dir / "epochs.csv").exists():
        changes.setdefault("training", {})["summary"] = training_summary(run_dir / "epochs.csv")
    if (run_dir / "eval.json").exists():
        changes["metrics"] = json.loads((run_dir / "eval.json").read_text())
    meta = registry.update_meta(run_dir, **changes)
    spec = meta["spec"]
    if "epochs" in meta.get("training", {}) and spec.get("steps_per_epoch"):
        meta["training"]["samples_per_epoch"] = int(spec["steps_per_epoch"] * spec["num_envs"])
        registry.write_meta(run_dir, meta)
    return meta
