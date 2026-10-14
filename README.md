# DronePPO: a 2-thruster drone learning to fly to targets

A stick-shaped drone with two thrusters (force and swivel angle each) flies to targets inside a walled world. Controllers are trained with PPO from scratch, or by
imitating a hand-written bang-bang controller and fine-tuning with PPO. Everything is one package, `drone_rl`.

## Commands (`python3 -m drone_rl <command>`)
| command | what it does |
|---|---|
| `view [run]` | open the viewer; flies the newest run toward the mouse. Left/Right switch runs, Tab opens the run menu, Space hides the HUD, Esc quits. `main_screen.py` does the same. |
| `batch specs/<file>.json` | train and evaluate every run in a specs file (`--parallel P --workers N`) |
| `eval <run> ...` / `eval --all` / `eval --missing` | evaluate runs on the standard courses |
| `imitate --name ... --group ... --tags ... --blurb ...` | behavior-clone the controller into a network (DAgger) |
| `baseline [--update]` | re-measure the controller and the first imitation net and warn when the numbers drift |
| `check-scheme <name> [params]` | sanity-check a reward scheme |
| `index` | rebuild `runs/index.json` |
| `report` | summary table and charts (`docs/reports/`) |
| `bench` | training-throughput benchmark |

Tests: `python3 -m pytest` (about 15 s: environment, every reward scheme in both world sizes, controller courses, a 2-epoch training run end to end).

## Layout
```
drone_rl/            the package
  config.py          physics constants (the world size is a per-run spec setting, default 2400x1600)
  physics.py drone.py   stick physics and the drone (thrusters, observations)
  env.py             DroneEnv, spec defaults (make_spec), numpy policy, env worker process
  trainer.py         PPO trainer (spec keys: init_from, freeze_obs_norm, critic_warmup_epochs, per-output init_log_std, angle_rate, ...)
  schemes/           reward systems, one file each (see schemes/README.md); add a file, name it in a spec
  controller.py      the hand-written bang-bang controller        imitation.py  copy it into a network
  evaluation.py      the standard evaluation: fixed + random courses, style, robustness metrics
  registry.py        run folders, meta.json, tags, index, favorites     runner.py  run preparation and finishing
  batch.py cli.py ...   commands          viewer/  the viewer and run menu
specs/               batch files (one file = one group of runs)
runs/                every run: runs/<date>_<name>/ with meta.json (+ index.json, favorites.json)
archive/             pre-refactor code and runs (legacy_runs are listed read-only in the menu)
docs/                FINDINGS.md (results and conclusions), ideas.md, reports/, baselines.json
tests/  tools/       tests, one-off scripts (tools/migrate_runs.py did the folder migration)
```

## Making a run
A spec (see `specs/b016.json`) needs a `name`, a `blurb` (`"short name\nwhat it tests, what changed from the branch point"`), a `group` (the branch family),
`tags`, and optionally a `parent` (the run it branched from). Everything else defaults (`drone_rl/env.py` DEFAULTS). Facts like scheme, world size, run kind
and gate tier are tagged automatically. Results land in `runs/<id>/meta.json`: spec, model size, training curve summary and ~80 evaluation metrics.

Note: `docs/FINDINGS.md` was written before the refactor; its run paths (`experiments/runs/<batch>/<name>`) are now `runs/<date>_<name>` (the old batch is the tag `batch:bXXX`),
and results measured in the 1200x800 world carry `world:1200x800`.
