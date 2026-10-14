"""Run registry: ids, meta round trip, index rows and tags, favorites, archive rows."""
import json
from datetime import datetime

from drone_rl import registry


def make_run(root, name, **meta_over):
    run_dir = registry.new_run_dir(name, datetime(2026, 1, 2, 3, 4), root)
    meta = {"id": run_dir.name, "name": name, "title": name.title(), "blurb_text": "does a thing", "group": "g", "parent": None, "tags": ["t1"], "kind": "ppo", "status": "ok",
            "created": "2026-01-02T03:04:00-05:00", "finished": "2026-01-02T03:10:00-05:00", "world": [1200, 800], "model": {"obs_dim": 17, "hidden": [16, 8], "params": 460},
            "spec": {"scheme": "time_shaping", "pi_lr": 3e-4, "params": {"k": 1}}, "legacy": {}, "training": {"epochs": 10, "wall_seconds": 60.0},
            "metrics": {"reaches_per_min": 12.5, "crash_rate": 0.0, "tier": 2, "nozzle_use_deg": 0.1}, **meta_over}
    registry.write_meta(run_dir, meta)
    return run_dir


def test_ids_are_dated_and_unique(tmp_path):
    a, b = make_run(tmp_path, "same"), make_run(tmp_path, "same")
    assert a.name == "2026-01-02_0304_same" and b.name == "2026-01-02_0304_same-2"


def test_find_run_by_name_id_and_prefix(tmp_path):
    d = make_run(tmp_path, "alpha")
    assert registry.find_run("alpha", tmp_path) == d and registry.find_run(d.name, tmp_path) == d and registry.find_run("2026-01-02_0304", tmp_path) == d


def test_index_rows_have_flat_metrics_and_tags(tmp_path):
    make_run(tmp_path, "alpha")
    (row,) = registry.build_index(tmp_path)
    assert row["reaches_per_min"] == 12.5 and row["hp.pi_lr"] == 3e-4 and row["param.k"] == 1 and row["model.params"] == 460 and row["train.epochs"] == 10
    assert {"t1", "kind:ppo", "scheme:time_shaping", "world:1200x800", "gates:2/3", "nozzles:neutral"} <= set(row["tags"])
    assert json.loads((tmp_path / "index.json").read_text())[0]["name"] == "alpha"


def test_favorites_toggle(tmp_path):
    assert registry.favorites(tmp_path) == []
    assert registry.toggle_favorite("x", tmp_path) == ["x"]
    assert registry.toggle_favorite("x", tmp_path) == []


def test_archive_rows_are_read_only(tmp_path):
    run = tmp_path / "runs" / "run3"
    run.mkdir(parents=True)
    (run / "actor.pt").write_bytes(b"x")
    (run / "blurb.txt").write_text("Short name\nWhat it did.\n")
    (run / "epochs.csv").write_text("epoch,episodes,mean_return,mean_length,total_reaches,pi_loss,v_loss,actor_lr,critic_lr\n" + "".join(f"{i},5,-1,50,{i},0,0,0.0003,0.001\n" for i in range(1, 21)))
    (row,) = registry.archive_rows(tmp_path)
    assert row["archived"] and not row["flyable"] and row["title"] == "run3: Short name" and row["train.epochs"] == 20 and row["curve.final_reaches_mean"] > 10
