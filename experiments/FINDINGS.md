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

## Note: target margin changed
After this search the target margin (min distance of a target from every wall) was changed from 100 px to 10 px in the training default, the fixed evaluation and the project config. All results above used the 100 px margin, so new runs are not directly comparable to them.

## World doubled to 2400x1600 (bang-bang session)
`core/config.py` ENV_WIDTH/ENV_HEIGHT are now 2400x1600. All results above (and runs/ policies) used 1200x800, so they are not comparable with new runs.
Bang-bang controller in the new world (36 episodes): thrust 1x -> 11.9 reaches/ep, 0% crashes, 310 px/s; 2x (default) -> 13.8, 3%, 377 px/s; 3x -> 14.1, 8%, 403 px/s; 5x -> 12.1, 25%. Throttling is limited by the 180-degree turns and the final approach, not by thruster force.

## Imitating the bang-bang controller, then PPO fine-tuning (b008-b011)
`bc_pretrain.py`: the controller flies (random start states, retargeting, action noise); a [24,16,8,4] network (scheme `bangbang_features`: 7 extra inputs - speed along/across the line to the target, braking margin, sin/cos of the body-angle error for accelerating and braking) is fit with a smooth-L1 loss; then DAgger rounds (the student flies, the controller labels the states it reaches). Action units: force_rate 0.5 (action 1 = 0.5 thruster force per step).
- Pure imitation (1.2 M samples, best DAgger round 3; later rounds got worse): 9.7 reaches/ep, 23% crashes, 321 px/s in the fixed evaluation (controller itself: 13.8 / 3% / 377). Saved at runs/bc/bc_feat.
- The policy is extremely sensitive to action noise: std 0.04 -> 0.8 reaches/ep, std 0.08 -> 0. PPO's default exploration (log_std -0.5, then -2.5) destroyed it within 25 epochs.
- PPO fine-tuning all failed to beat the imitation start: b008 (pi_lr 1e-4 / 3e-5, loose-to-strict reach curriculum, log_std -2.5): collapsed to 0 reaches by epoch 150 (stopped, runs deleted). b009/b010 (strict reach rule from the start; log_std -2.5 / -5): 0 reaches within 50 epochs (stopped). b011 (log_std -4, pi_lr 1e-5 / 3e-6, clip 0.1, 2 update epochs, 20 critic warm-up epochs, 100 epochs): evaluated 2.7 reaches/ep, 73% crashes and 1.9 reaches/ep, 63% crashes - worse than the imitation start.
- Cause (not fully isolated): with tiny exploration noise the probability ratios swing wildly, so the PPO steps are noisy and each step erodes the sharp imitated behaviour.

## Why PPO fine-tuning made the imitation network worse (b012-b014)
Cause: the trainer kept updating the observation normalizer (running mean/std) during fine-tuning, so the imitation network was fed inputs scaled differently from the ones it was trained on. Visible with the actor frozen (lr 0): reaches per epoch fell from 126 to ~65 with the normalizer updating (b012 t1_norm_free) and stayed ~160 with it frozen (t1_norm_frozen). New spec key `freeze_obs_norm`.
With the normalizer frozen (log_std -4, clip 0.1, 2 update epochs, 15 critic warm-up epochs, 100 epochs), fixed evaluation, 30 episodes: pi_lr 1e-4: 10.7 reaches/ep, 40% crashes, 357 px/s; pi_lr 1e-5: 10.6, 37%, 352 (imitation start: 9.7, 23%, 321). Reaches +9%, speed +10%, crashes up. Crash penalty 30000: 10.0, 33%. log_std -6 (almost no exploration): 9.9, 23% (stays at the imitation level).
Other suspected causes: the critic warm-up claim made here earlier relied on b011, which had the normalizer bug, so it was retested (b015, normalizer frozen, pi_lr 1e-4, 150 epochs): warm-up 0 / 15 / 60 epochs gave 11.4 / 11.1 / 10.7 reaches per episode and 30% / 33% / 40% crashes (30 episodes each, so within noise). A longer critic warm-up does not help. Exploration noise costs ~25% of training reaches at std 0.018 but is not what made it collapse. Not tested: a BC-anchor loss, KL early stopping.

## Critic warm-up retest (b015)
With the normalizer frozen and pi_lr 1e-4: warm-up 0 -> 11.4 reaches/ep, 30% crashes, 365 px/s; 15 -> 11.1, 33%, 363; 60 -> 10.7, 40%, 366. No benefit from warming the critic longer (it leaves fewer PPO epochs, and the trend is slightly worse).

## Can the nozzles help? (controller vectoring test, b016)
Geometry check: tilting both nozzles the same way (one outward, one inward, equal push across the stick) turns the thrust sideways with zero torque (checked in the physics: 30 deg tilt gives lateral/vertical 300/520 px/s^2). Controller with thrust vectoring (`vector_deg`, body turns only the rest of the way), 36 episodes, world 2400x1600: neutral 14.3 reaches/ep, 0% crashes; 20 deg 13.6, 11%; 35 deg 11.5, 25%; 50 deg 10.2, 36%. Single-leg traces: settle times equal or slightly worse (102 vs 102, 107 vs 113, 96 vs 100 steps). The 180-degree turns take about 0.12 s, so turning time is not what limits the controller; nozzle steering has little to save.
PPO with exploration noise on the nozzle outputs (from the imitation net, normalizer frozen, pi_lr 1e-4, 100 epochs): angle log std -1 -> 11.2 reaches/ep, 23% crashes, nozzle outputs stay ~0 (PPO found no use for them); same with nozzles 6x faster per action (angle_rate 0.3) -> 8.6, 37%, nozzle outputs dither (std 1.2, +-20 deg) with no benefit. New spec keys: `angle_rate`, list-valued `init_log_std` (per output).
