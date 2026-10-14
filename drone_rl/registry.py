"""The run registry: every run lives in runs/<YYYY-MM-DD_HHMM>_<name>/ with a meta.json that is the single source of truth about it.
meta.json holds the name, blurb, group (branch family), parent run, tags, timestamps, world size, model size, the full spec, training summary and evaluation metrics.
runs/index.json is a generated flat table of all runs (rebuilt by `python3 -m drone_rl index`); runs/favorites.json holds the pinned run ids."""
import json
import math
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
ARCHIVE = REPO / "archive" / "legacy_runs"
META = "meta.json"
REQUIRED_SPEC_FIELDS = ("name", "blurb", "group", "tags")  # a spec for a new run must carry these (see check_run_spec)


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")


def run_id(name, when):
    return f"{when.strftime('%Y-%m-%d_%H%M')}_{slug(name)}"


def new_run_dir(name, when=None, root=None):
    """Creates and returns runs/<id>/ (a numeric suffix is added if a run with that id already exists)."""
    root = Path(root or RUNS)
    when = when or datetime.now()
    base = run_id(name, when)
    path, n = root / base, 1
    while path.exists():
        n += 1
        path = root / f"{base}-{n}"
    path.mkdir(parents=True)
    return path


def check_run_spec(spec):
    """Raises ValueError unless a spec for a new run has a name, a blurb ('short name' line, then 1-2 short sentences), a group and at least one tag."""
    missing = [k for k in REQUIRED_SPEC_FIELDS if not spec.get(k)]
    if missing:
        raise ValueError(f"run {spec.get('name', '?')!r} is missing {missing}: every run needs a name, a blurb ('short name\\nwhat it tests, what changed from the branch point'), a group and tags")
    if not isinstance(spec["tags"], (list, tuple)):
        raise ValueError("tags must be a list of strings")
    lines = spec["blurb"].strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"run {spec['name']!r}: the blurb needs two lines, a short name and then the description")


def split_blurb(blurb):
    """('short name', 'description') from the two-line blurb text."""
    lines = [l.strip() for l in str(blurb).strip().splitlines() if l.strip()]
    return (lines[0] if lines else ""), " ".join(lines[1:])


def read_meta(run_dir):
    return json.loads((Path(run_dir) / META).read_text())


def write_meta(run_dir, meta):
    (Path(run_dir) / META).write_text(json.dumps(meta, indent=1, default=_json_default) + "\n")


def update_meta(run_dir, **changes):
    meta = read_meta(run_dir)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(meta.get(key), dict):
            meta[key].update(value)
        else:
            meta[key] = value
    write_meta(run_dir, meta)
    return meta


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj)}")


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def kind_of(spec):
    """What produced the run: 'controller' (hand-written), 'imitation' (supervised copy of the controller), 'ppo-finetune' (PPO from a pretrained network) or 'ppo'."""
    if spec.get("kind"):
        return spec["kind"]
    return "ppo-finetune" if spec.get("init_from") else "ppo"


def auto_tags(meta):
    """Tags that are facts about the run, computed from its spec and metrics."""
    spec = meta.get("spec", {})
    tags = [f"kind:{meta.get('kind', kind_of(spec))}", f"scheme:{spec.get('scheme', '?')}"]
    world = meta.get("world")
    if world:
        tags.append(f"world:{int(world[0])}x{int(world[1])}")
    if spec.get("init_from"):
        tags.append("pretrained-start")
    if spec.get("freeze_obs_norm"):
        tags.append("frozen-normalizer")
    metrics = meta.get("metrics", {})
    if metrics.get("tier") is not None:
        tags.append(f"gates:{int(metrics['tier'])}/3")
    nozzle_use = metrics.get("nozzle_use_deg")
    if nozzle_use is not None:
        tags.append("nozzles:moving" if nozzle_use > 3.0 else "nozzles:neutral")
    if meta.get("legacy", {}).get("batch"):
        tags.append(f"batch:{meta['legacy']['batch']}")
    return tags


def all_tags(meta):
    seen, out = set(), []
    for tag in list(meta.get("tags", [])) + auto_tags(meta):
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def iter_runs(root=None):
    """(run_dir, meta) for every run folder that has a readable meta.json, oldest first."""
    root = Path(root or RUNS)
    if not root.exists():
        return []
    found = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / META).exists():
            try:
                found.append((d, read_meta(d)))
            except (OSError, ValueError):
                continue
    return found


def find_run(ref, root=None):
    """Run directory for a run id, an id prefix, or a unique name. Raises KeyError if nothing or more than one matches."""
    root = Path(root or RUNS)
    if (root / ref / META).exists():
        return root / ref
    hits = [d for d, m in iter_runs(root) if m.get("name") == ref or d.name.startswith(ref) or d.name.endswith("_" + ref)]
    if len(hits) != 1:
        raise KeyError(f"{ref!r} matches {len(hits)} runs")
    return hits[0]


def index_row(run_dir, meta):
    """One flat row for the index: identity, grouping, timing, size, hyperparameters and every metric."""
    spec = meta.get("spec", {})
    row = {"id": Path(run_dir).name, "name": meta.get("name"), "title": meta.get("title"), "blurb": meta.get("blurb_text"), "group": meta.get("group"),
           "parent": meta.get("parent"), "kind": meta.get("kind"), "status": meta.get("status"), "created": meta.get("created"), "finished": meta.get("finished"),
           "tags": all_tags(meta), "world": meta.get("world"), "flyable": meta.get("flyable", True), "archived": meta.get("archived", False)}
    row.update({f"model.{k}": v for k, v in meta.get("model", {}).items()})
    row.update({f"train.{k}": v for k, v in meta.get("training", {}).items() if not isinstance(v, (dict, list))})
    row.update({f"curve.{k}": v for k, v in meta.get("training", {}).get("summary", {}).items()})
    row.update({f"hp.{k}": spec[k] for k in sorted(spec) if k not in ("params", "blurb", "tags")})
    row.update({f"param.{k}": v for k, v in (spec.get("params") or {}).items()})
    row.update({k: v for k, v in meta.get("metrics", {}).items() if isinstance(v, (int, float, str, bool)) or v is None})
    return row


def archive_rows(root=None):
    """Index rows for the pre-refactor runs in archive/legacy_runs: read-only (they use the old observation layout), so only what their files show: blurb, epochs, last reach counts."""
    from datetime import datetime
    root = Path(root or ARCHIVE)
    rows = []
    if not root.exists():
        return rows
    for family in sorted(p for p in root.iterdir() if p.is_dir()):
        for d in sorted((p for p in family.iterdir() if p.is_dir() and (p / "actor.pt").exists()), key=lambda p: (len(p.name), p.name)):
            title, text = (d.name, "No description recorded.")
            if (d / "blurb.txt").exists():
                t, tx = split_blurb((d / "blurb.txt").read_text())
                title, text = f"{d.name}: {t}", tx
            stamp = (d / "epochs.csv").stat().st_mtime if (d / "epochs.csv").exists() else (d / "actor.pt").stat().st_mtime
            row = {"id": f"archive/{family.name}/{d.name}", "name": d.name, "title": title, "blurb": text, "group": "legacy runs (original trainer)" if family.name == "runs" else f"legacy runs ({family.name})",
                   "parent": None, "kind": "ppo", "status": "archived", "created": None, "finished": datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds"),
                   "tags": ["legacy", "archive", f"folder:{family.name}"], "world": None, "flyable": False, "archived": True}
            if (d / "epochs.csv").exists():
                from drone_rl.runner import training_summary
                try:
                    summary = training_summary(d / "epochs.csv")
                    row.update({"train.epochs": summary.get("epochs_logged"), "curve.final_reaches_mean": summary.get("final_reaches_mean"), "curve.best_reaches": summary.get("best_reaches"),
                                "curve.final_return": summary.get("final_return"), "curve.final_episode_length": summary.get("final_episode_length")})
                except (KeyError, ValueError):
                    pass
            rows.append(row)
    return rows


def build_index(root=None):
    """Writes runs/index.json (all runs plus the read-only archive, flat rows, newest first) and returns the rows."""
    root = Path(root or RUNS)
    rows = [index_row(d, m) for d, m in iter_runs(root)]
    if root == RUNS:
        rows += archive_rows()
    rows.sort(key=lambda r: r.get("created") or r.get("finished") or "", reverse=True)
    (root / "index.json").write_text(json.dumps(rows, indent=1, default=_json_default) + "\n")
    return rows


def load_index(root=None):
    path = Path(root or RUNS) / "index.json"
    return json.loads(path.read_text()) if path.exists() else build_index(root)


def favorites(root=None):
    path = Path(root or RUNS) / "favorites.json"
    return json.loads(path.read_text()) if path.exists() else []


def toggle_favorite(run, root=None):
    root = Path(root or RUNS)
    favs = favorites(root)
    favs = [f for f in favs if f != run] if run in favs else favs + [run]
    (root / "favorites.json").write_text(json.dumps(favs, indent=1) + "\n")
    return favs


def number_params(hidden, obs_dim, act_dim=4):
    sizes = [obs_dim, *hidden, act_dim]
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def safe_float(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None
