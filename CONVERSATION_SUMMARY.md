# DronePPO_iarcTesting: conversation summary (turn by turn)

Written so the conversation can be compacted. The turn order is reconstructed from memory of the conversation, so small details may be slightly off.
Numbers quoted are from the runs described, usually from a headless evaluation (mean action, no exploration noise).

## Project at a glance

- `main_screen.py`: pygame viewer. Loads a saved actor (`RUN_INDEX`), flies a drone toward the mouse, draws the network top-left and the world bounds. Set `RUN_INDEX = "12/snapshots/epoch2000"` style strings to load snapshots (untried in the real window).
- `ppo.py`: headless PPO from scratch (PyTorch). Classes: `Actor`, `Critic`, `RunningNorm`, `NumpyPolicy`, `DroneEnv`, `SubprocEnvs`/`env_worker` (one process per environment), `EpisodeHandler`, `TrainHandler`. CLI: `--model`, `--epochs`, `--steps-per-epoch` (per environment), `--num-envs`, `--episode-length`, `--resume RUN`, `--ramp-epochs`, `--lr-decay`, `--lr-schedule`.
- `drone.py`: the drone physics and observation (copied from `DronePPO`, then heavily changed). `config.py`: every simulation and reward constant. `network.py`, `calcs.py`, `text.py`, `fontDict.py`: helpers.
- Run folders: `runs/` (current), `runsPreAngleControl/` (run1-7, 16-input / 2-output era), `runsPreLevelEncouragement/` (run1-2). Each run folder holds `actor.pt`, `critic.pt`, `norm.npz` (observation/reward statistics, needed to fly a model), `epochs.csv`, `episodes.csv`, `rewards.png`, and optionally `snapshots/epoch<N>/`.
- The evaluation scripts I used live only in `/tmp` (`evalrun.py`, `evalrun40.py`, `evalang.py`, `dist_eval.py`, `asym.py`, `equiv.py`, `mirror.py`, `time_new.py`) and may be gone. They measure: speed, mean tilt, reaches and episode length; thruster angle statistics; reach rate by target distance; left-to-right vs right-to-left flight smoothness; policy mirror-symmetry.

## Turn by turn

### Part 1: the pygame viewer and network.py
1. **Why does main run at 40 fps?** (no edits asked) There is no `main.py`; `main_screen.py` caps at 60. The frame cost comes from three full-resolution `transform.scale` calls plus alpha blits at fullscreen.
2. **Compare with other pygame projects.** About 40 projects use fullscreen and most cap at 60. `Framework/main_screen.py` is identical to this loop, and the environment (arm64 Python 3.13, pygame 2.6.1) is normal. A headless benchmark showed the scaling and blits cost roughly 5 ms + 4 ms at 1512x982 and 18 ms + 15 ms at 3024x1964.
3. **Survey more projects, and what resolution for a 3x downscale?** The desktop reports 1800x1169. Suggested 1512x984 (divisible by 3) or `pygame.SCALED`.
4. **Remove all scaling logic, edit only `main_screen.py`.** Done (fixed fonts, direct blits, mouse coordinates, cursor radius 5). I introduced and then fixed a parenthesis bug in the FPS label placement.
5. **Look through `network.py`.** It was already numpy and isn't imported by `main_screen.py`; the useful rewrite target was `draw_network`, plus small numeric wins.
6. **Rewrite `network.py`; how to get rid of a macOS message.** Rewrote it (pixel-identical output verified). The `ApplePersistenceIgnoreState` line is harmless; `defaults write org.python.python ApplePersistenceIgnoreState NO` silences it.
7. **Init a network in main and draw it.** Added a random 2-3-3-1 network.

### Part 2: drone and first PPO
8. **Copy `drone.py` from `DronePPO`.** Copied; it needed `config` and `calcs` helpers.
9. **"yes" (copy `config.py` and add helpers).** Copied `config.py`, added `clamp` and `normalize_angle_signed` to `calcs.py`.
10. **Init a drone in main, falling.** Fixed 1/50 s physics step with an accumulator, respawn below the screen.
11. **Write `ppo.py` with `RolloutHandler` and `TrainHandler`.** First torch skeleton with placeholder env.
12. **Bare-bones PPO spec + /grillMe.** Decisions: world 1000x800; rollout = one episode, epoch = steps across several episodes; target margin 100 px; reach = within 20 px and under 50 px/s; reward `-dist/diag` plus +10 per reach; random `Drone.reset()` start; keep the 16-value observation; install matplotlib. A fall test showed the drone leaves the screen in about 63 steps. Epoch size set to 1024, episode timeout 512.
13. **"yes": build and run.** Rewrote `ppo.py` minimally with run folders, CSVs and a plot. First 50-epoch run did not learn.
14. **"do the first 3".** Crash penalty, a positive per-step reward (`1 - dist/diag`), and 200 epochs: the drone learned to survive the full 512 steps but never reached targets.
15. **Does a positive per-step reward change anything?** Explained: it does, because a crash ends the episode (terminal value 0), so a constant shift changes the incentive to crash.
16. **Loader function in main.** Added `load_run` with a hardcoded `RUN_INDEX`.
17. **Draw the actor network small, top-left; mouse is the target.** Done.
18. **Train 16x16x8x2 for 500 epochs.** Learned slowly.
19. **Revert the mouse dot.** Cursor radius back to 2.
20. **Make `ppo.py` a terminal command.** Added argparse (`--model`, `--epochs`, `--episode-length`) and live progress lines.

### Part 3: rewards, normalization, exploration noise
21. **What other rewards/changes help?** Listed ideas (progress/velocity rewards, tilt and spin penalties, hover start, observation and reward normalization, bigger batches, KL guard).
22. **Chosen: velocity (cosine) reward, zone reward, hover-start thrusters. Questions on unnormalized features, reward scales, `log_std`.** Implemented and answered. Key point: the sampled action std (0.61) was far larger than the env's +/-0.10 clip range.
23. **Explain `log_std` in depth; what "env clipping" means; can we just scale the policy output; what if we remove clipping.** Explained that clipping hid most of the noise and made the mean nearly irrelevant.
24. **"do this": remove clipping and lower the rates 10x.** Force rate 0.005, angle rate pi/60 per unit action; no `np.clip`.
25. **Rewrite `config.py` to only the used rewards.** Removed the old reward system; reward constants now live in `config.py`.
26. **Rename to `EpisodeHandler`; rescale the velocity feature (divisor 300); remove the angle feature; velocity reward 0.5; default model 15x32x32x2.** Done.
27. **NaN mean return for some epochs.** Cause: epochs where no episode finished (episode longer than the epoch).
28. **`--steps-per-epoch` (default 2048), printout shows the unfinished episode with `*`, checkpoints every 20 epochs.** Done.
29. **Why does the drone reset in main?** The 10 s time limit, or leaving the 1000x800 world.
30. **Limit of 500 s; draw the bounds.** Done (world fill switched to blue-grey so the black outline shows).

### Part 4: thruster angle control and the outward posture
31. **Two more outputs: thruster direction.** Ported the idea from `EvolutionAlgorithmV2DroneControl`; 4 actions (2 force changes, 2 angle changes), angle observations, 17 inputs. Verified limits and hover.
32. **Rename `runs` to `runsPreAngleControl`, new `runs`.** Done.
33. **Reward for pointing thrusters outward; why did early training pin the angles?** Added a reward peaked at 30 degrees outward. Analysis: rate-controlled angle + noise much larger than the action range -> saturation.
34. **Resume option, double the outward reward, penalty for being non-level for long periods.** Added `--resume RUN` (creates `run2.1`-style folders), outward reward 1.0, a tilt penalty whose weight ramps up slowly and falls quickly.
35. **Separate learning rates, lr decay (exponential to 0.33), tilt weight as an input (18 inputs); new runs folder, old one renamed `runsPreLevelEncouragement`.** Done (actor 3e-4, critic 1e-3).
36. **Run a 500-epoch session.** First run where the drone reached targets.
37. **The drone wobbles: add a level reward; epoch 4096 steps, episode 2048; 100 epochs.** Added the level reward with a ramp; training survived but wobble persisted.
38. **Halve the tilt penalty, double the level reward, ramps 3x longer; other ideas?** Applied; listed ideas (angular-velocity penalty, action-change penalty, entropy, KL, etc.).

### Part 5: normalization, parallel environments, speed
39. **Angular-velocity penalty, fewer update epochs (4), normalize observations and rewards, 8 parallel environments.** Implemented (running mean/std, clip 10; reward scaled by return std). Gave a numerical example of 8 environments.
40. **Run on separate cores; scale steps per epoch.** Worker processes via `SubprocVecEnv`; `--steps-per-epoch` became steps per environment (default 4096 -> 32768 total); minibatches fixed at 16.
41. **Other speed-ups; run a 200-epoch session in the background.** Run3.
42. **Do speed items 1 and 2; more workers vs several per core; asynchronous workers.** Simulator step 54 us -> 17 us (checked equivalent to the old simulator, max obs difference 6e-5). Workers now hold a numpy copy of the policy and run a whole stretch of steps alone (about 160,000 steps/s at 8 workers vs 12,500). Scaling: 4 workers 88k, 8 -> 160k, 10 -> 174k, 16 -> 99k steps/s on 10 cores (4 P + 6 E).
43. **Run3 learned to move extremely slowly and stay level: rebalance rewards.** Diagnosis: standing rewards (~3.7/step) dwarfed the +10 reach bonus. New balance: reach 500, crash 100, distance 0.3, level 0.5, outward 0.3, velocity reward = approach speed / 100 px/s. Run4: 224 reaches per epoch at 100 epochs; flies 168 px/s but tilts 25 degrees.
44. **Do the dials; thrusters aren't going outward.** Level 1.0, tilt cap 3.0, crash 300; outward reward made steeper (0 at neutral, full at 30 degrees, negative inward). Run5/run6: still not outward (mean outward reward negative); policy uses thruster angles to steer.
45. **Option 1: outward-only thruster range, continue run6 for 300 epochs, run in a background task.** Run6.2: much worse (tilt 39, episodes 351 steps) - the constraint removed steering authority (interrupted first attempt left an empty folder, removed).
46. **Widen to -15 degrees inward / 60 outward, fresh 500 epochs.** Run7: level (3.4 degrees tilt), 14 reaches per episode, outward posture not learned (both outward 5% of steps).
47. **Outward x10 plus a ramp that relaxes level/tilt over the session; 300 epochs.** Run8: thrusters outward 100% of steps but body tilt 51 degrees, 4.7 reaches per episode.
48. **Range -15..+90, outward 2.0, tilt penalty up; resume 200 epochs.** Run8.1: still outward, tilt 58 degrees, reaches 6.8 per episode.
49. **Outward 1.0, inward 30 degrees, extra tilt penalty when slow; fresh 500 epochs.** Run9: best flyer so far (190 px/s, 4.8 degrees tilt, 17 reaches per episode) but outward posture lost. Conclusion: outward posture trades off against flight performance.

### Part 6: bigger world, symmetry, long runs
50. **Outward 1.5; "zoom out" (bigger world); far-target curriculum; 128x128x128; 1000 epochs.** World 1500x1000; far targets (>=600 px) grow to 70% over the first half; run10: 100% reach in every distance bin, 363 px/s, 43 degrees tilt, return still rising slowly (no plateau-break).
51. **Flies smoothly left-to-right, wobbles right-to-left.** Verified the environment is exactly mirror-symmetric (observations, rewards, positions identical); the asymmetry was learned (run9 even tumbled right-to-left). Added random per-episode mirroring (50%) and symmetric observation statistics, plus three small smoothness penalties (action change, angular acceleration, lateral velocity) phased in between 30% and 70% of a session. Resumed run10 for 300 epochs (run10.1): asymmetry reduced (right-to-left angular speed 4.3 vs 1.8 rad/s) but not removed; policy still ~99% asymmetric in a direct test.
52. **Symmetric-network proposal** (policy/critic averaged with their mirror). Proposed, **not implemented**.
53. **Reach reward x2 (1000), episodes x2 (4096), overnight session.** Added `--ramp-epochs`, append-only CSV writing, 500-epoch snapshots, keep-awake (`caffeinate -i`). Stricter reach rule requested mid-turn: speed < 20 px/s, angular speed < 0.5 rad/s (both start looser and tighten over the curriculum), plus a hold of 25 consecutive calm steps (ramped from 1). Short run11 (300 epochs) learned fine (175 reaches/epoch). Estimated 75,000 epochs for 10 hours at ~0.44-0.48 s/epoch.
54. **Overnight run12 results.** All 75,000 epochs took 9.06 hours. The best models were early: epoch 1000-2000 snapshots (~12-13 reaches/episode, near-symmetric: angular speed 0.58 vs 0.61 rad/s). Performance degraded between epochs 3,000 and 50,000 (as low as 5-6 reaches/episode) and only partly recovered; the final model is the smoothest but crashes more on 900+ px targets. Likely cause (untested): a nearly constant learning rate for thousands of epochs.
55. **New 2000-epoch session, lr decay to 0.01.** Run13: stable, 100% reach in every distance bin including 1200-1800 px, symmetric (angular speed 0.64 vs 0.42 rad/s), ~12 reaches per episode with 40-episode tests; no late collapse.
56. **Model 18x16x16x4, 300 epochs.** Run14 with exponential decay to 0.01 over 300 epochs: zero reaches. A control (run15, same model, decay to 0.33) learned to reach (5.5 reaches/episode, 100% up to 900 px), so the fast decay (not model size alone) was the main cause.
57. **Default model 18x16x8x4 (user wrote x2; outputs are 4), default epochs 400; what about linear lr decay?** Defaults changed; explained linear vs exponential.
58. **"yes": add a `--lr-schedule` option, linear to 0.1 as default; write this summary.** Added; run16 started with the new defaults (results below if filled in).

## Run index (in `runs/`, unless noted)

| Run | What | Key result |
|---|---|---|
| run1 | 500 epochs, first session after the level-encouragement folder reset | first real reaches |
| run2 | 100 epochs, level reward, 2048-step episodes | survival, wobble persisted |
| run3 | 200 epochs, async/normalization era | learned to move very slowly and stay level (21 px/s, 0.7 deg) |
| run4 | 100 epochs, rebalanced rewards | 168 px/s, 25 deg tilt, 12.8 reaches/episode |
| run5 | 100 epochs, steeper outward reward | tilt 32 deg, outward ignored |
| run6 | 500 epochs | 18.6 reaches/episode, outward ignored |
| run6.2 | run6 + 300, outward-only limits | much worse |
| run7 | 500, -15..+60 range | level (3.4 deg), 14 reaches |
| run8 / 8.1 | outward 5.0 / 2.0 | outward posture, 51-58 deg tilt |
| run9 | 500, outward 1.0, slow-tilt penalty | best flyer of that era |
| run10 / 10.1 | bigger world, 128^3, curriculum / + mirroring | 100% reach in all bins; asymmetry reduced |
| run11 | 300, stricter reach + hold | learned fine |
| run12 | 75,000 epochs overnight | best early, degraded, partial recovery; snapshots kept |
| run13 | 2000 epochs, lr to 0.01 | stable and good, symmetric |
| run14 / run15 | 16x16 model, lr to 0.01 / 0.33 | failed / learned |
| run16 | defaults: [18,16,8,4], 400 epochs, linear lr to 0.1 | see results below |

## Current settings (end of conversation)

- World 1500x1000; targets spawn >=100 px from edges; far-target curriculum 0 -> 70% (>=600 px) over the first half of the session.
- Observation: 18 values (17 drone values + tilt weight); 4 actions. Actions are not clipped; rates 0.005 (force) and pi/60 (angle) per unit action. Thruster angle limits: left -30..+90 degrees, right -90..+30 (outward positive).
- Rewards (per step): distance 0.3, approach speed (clipped, /100 px/s) 1.0, zone 0.5, outward 1.5 (peak at 30 degrees), level reward cap 1.0 -> 0.25 and tilt penalty cap 4.0 -> 2.0 (relax by half the session), slow-tilt multiplier, angular velocity penalty 0.1 beyond 0.3 rad/s, smoothness penalties (action change 0.2, angular accel 0.2, lateral velocity 0.3; 30%-70% ramp); reach +1000 (within 20 px, speed < 20 px/s, spin < 0.5 rad/s, held 25 steps; limits start looser and ramp); crash -300.
- Training: PPO, gamma 0.99, GAE 0.95, clip 0.2, 4 passes, 16 minibatches, 8 environments x 4096 steps per epoch, actor lr 3e-4 / critic 1e-3, linear decay to 0.1x, checkpoint every 20 epochs, snapshot every 500; defaults: model [18,16,8,4], 400 epochs, episode length 4096, random mirroring on.

## Findings worth remembering

- A reward balance problem (standing rewards >> reach bonus) produced "hover forever" policies; the reach bonus has to be worth many steps of standing rewards.
- Outward thrusters trade off against flight skill; no weight tried got both. Outward-only limits removed steering authority.
- The environment is exactly mirror-symmetric; asymmetric flight is learned. Random mirroring from the start fixed most of it; resuming an asymmetric policy only reduced it.
- Very long runs at a nearly constant learning rate degraded (run12). Decaying the learning rate fixed that (run13). A too-fast decay in a short session starves learning (run14); a gentler or linear decay works (run15).
- Performance comparisons across runs are loose: evaluations use 12-40 episodes, and many changes were made together.

## Open items

- Symmetric network (policy and critic averaged with their mirror) was proposed but not implemented.
- Outward posture and level/fast flight remain a trade-off; hover quality after reaching was only checked via the 25-step hold rule.
- Snapshot loading in `main_screen.py` (`RUN_INDEX` as a path string) has not been tried in the real window; the eval scripts in `/tmp` could be moved into the project.

## Run16 results (default settings, 400 epochs, linear lr to 0.1)

Run16 = the new defaults ([18,16,8,4], 400 epochs, 8 envs x 4096 steps, episode length 4096, linear lr 3e-4 -> 3e-5 for the actor). It learned to reach but is weaker than the bigger models and than run15.

Training windows (mean return / mean length of 4096 / reaches per epoch):
- epochs 1-50: -264 / 118 / 0.5
- epochs 51-150: 3,327 / 1,770 / 36.0
- epochs 151-250: 6,713 / 2,279 / 80.2 (best window)
- epochs 251-325: 5,092 / 1,938 / 77.1
- epochs 326-400: 5,580 / 1,905 / 87.2

Flight test (40 episodes, 2048 steps, mean action): 151.6 px/s, mean tilt 26.2 degrees, 4.5 reaches per episode, episode length 1548.
Reach by distance (600 trials from rest): 0-300 px 100%; 300-600 px 98.7%; 600-900 px 92.0% (8% left the world); 900-1200 px 33.8% (66% left the world); 1200-1800 px 7.7% (13 trials).
Left/right test (900 px flights): left-to-right reached 0/3, right-to-left reached 2/3; angular speed 0.47 vs 0.80 rad/s, angle std 19.5 vs 29.5 degrees, so some asymmetry has returned with the small model.

Compared with the other small-model run (run15: [18,16,16,4], exponential lr to 0.33, 300 epochs; 5.5 reaches per episode, 100% up to 900 px), run16 is slightly worse, with more far-target failures. Possible reasons (untested): the smaller 16x8 network, linear decay ending at 0.1 vs 0.33 giving a different late-training lr, or run-to-run noise from a single run each.

Ideas not yet tried for next steps: a larger small model (e.g. [18,32,32,4]), more epochs with the linear schedule, the symmetric network, or comparing several seeds before drawing conclusions.
