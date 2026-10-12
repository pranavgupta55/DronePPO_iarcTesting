# Overnight reward search: findings

169 sessions (300 epochs each, 1200x800 world, network [17+extra,16,8,4], strict reach rule), 169 finished, **14 pass all three gates** (crash < 2%, no hover tilt / tilt streaks / rocking, both nozzles outward > 70% of steps). All tables, charts and run folders: `results.csv`, `reports/`, `runs/<batch>/`, `logs/`. Scheme catalogue: `schemes/catalog_*.md`, idea list: `ideas.md`.

## Winners (3 seeds each; reaches per 2,048-step episode)
| Scheme | Params (spec) | Seeds | Mean | Gates passed |
|---|---|---|---|---|
| `time_to_wall_barrier`, strong weights | `{"approach_w": 40.0}`, `outward_w=30` | 8.63 / 7.07 / 8.57 | **8.1** | 2 of 3 seeds (one had a 3.3% crash rate) |
| `curriculum_time_baseline`, strong weights | `{"base_k": 1600.0, "approach_w": 160.0}`, `outward_w=30` | 8.10 / 6.73 / 6.77 | 7.2 | 3 of 3 |
Reference: plain `time_shaping` (weight 2000) gave 7.0 and 0.4 reaches on two seeds and never points the nozzles outward.

## What the search showed
1. **Dense reward magnitude was the main lever.** At the default weights almost every new scheme hovered (best 0.6 reaches/episode). Multiplying each scheme's main progress weight by 10-160x (b002) brought many to 5-8 reaches/episode; the best scores sat at the top multiplier, so even stronger weights may help.
2. **Outward nozzles need a strong, separate term.** `outward_w` of 3 changed nothing (0% outward); 30 gave 97-99.9% outward with no loss of speed in the best schemes (b003); an earlier test at 20 on plain `time_shaping` made it spin. The overlay is the `outward_w` / `outward_target` spec key (added to `core/env.py`).
3. **The tilt/wobble gates were rarely binding** once the drone reached targets: nearly every run that learned to fly passed hover-tilt, tilt-streak and rocking limits (one seed of `time_to_wall_barrier_x160` failed hover tilt at 0.25).
4. **Seed noise is large** (same settings: 0.4 vs 7.0 reaches). Only gaps above about 2x should be trusted.
5. **Throughput on this machine:** all-background priority (`taskpolicy -b`), 2 sessions x 2 workers ran 300 epochs in about 2 min and was quiet; 3-4 sessions oversubscribed the 6 efficiency cores and slowed 4-6x; a 6 efficiency + 2 performance split also slowed runs, since workers wait on the slowest at every epoch.

## Not done / next
- Weights above the sweep range (x640+), outward weights between 3 and 30, and joint search of the top 2-3 schemes' secondary parameters.
- Untried: learning-rate, epoch-length, target-spawn and far-target settings were left at defaults.
- Gate thresholds were never recalibrated against trained policies; tilt gates may be too loose.
- `results.csv.bak` is the file before the column repair (two columns were added for the outward overlay).
