"""A 2-epoch training run completes, writes a loadable run folder with meta.json, evaluates, and shows up in the index. Takes about a minute."""
import json

from drone_rl import registry, runner
from drone_rl.evaluation import evaluate_run
from drone_rl.trainer import train


def test_train_evaluate_register(tmp_path):
    spec = {"name": "smoke", "scheme": "time_shaping", "epochs": 2, "num_envs": 1, "steps_per_epoch": 512, "episode_length": 256, "world": [1200, 800],
            "blurb": "Smoke test\nTwo epochs to check the pipeline.", "group": "tests", "tags": ["smoke"], "num_minibatches": 2, "update_epochs": 1}
    run_dir, full = runner.prepare_run(spec, root=tmp_path)
    train(full, run_dir)
    metrics = evaluate_run(run_dir, procs=2)
    meta = runner.finalize_run(run_dir)
    assert meta["status"] == "ok" and meta["world"] == [1200, 800]
    assert meta["training"]["epochs"] == 2 and meta["training"]["summary"]["epochs_logged"] == 2
    assert "reaches_per_min" in metrics and "fixed.long_legs.reached_frac" in metrics and "robust.noise05.crash_rate" in metrics
    rows = registry.build_index(tmp_path)
    assert len(rows) == 1 and rows[0]["name"] == "smoke" and "scheme:time_shaping" in rows[0]["tags"]
    assert json.loads((run_dir / "spec.json").read_text())["world"] == [1200, 800]
