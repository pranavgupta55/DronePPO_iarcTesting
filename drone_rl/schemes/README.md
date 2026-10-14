# Reward scheme plugins

A scheme is a module `drone_rl/schemes/<name>.py`. The spec's `"scheme": "<name>"` and `"params": {...}` select it; workers import it by name, so it must be
importable (`from schemes.x import ...` works) and its `Scheme` constructible from `params` alone. Params missing from the spec should fall back to
defaults kept in the module (`DEFAULTS` dict). Validate with `python3 -m drone_rl check-scheme <name> '{"param": 1}'`.

## Module contents
- `NAME` (str)
- `EXTRA_OBS` (int 0..7): number of extra observation features appended after the drone's 17 base observations.
- `EXTRA_OBS_MIRROR` (list of +1/-1, length EXTRA_OBS): multiplier applied to each extra feature when the episode is mirrored (left/right flipped). Use -1 for
  quantities that flip sign under a left-right mirror (x-velocity, angle, spin, ...), +1 for symmetric ones. Extras are computed in the physical
  (unmirrored) frame; the env mirrors them for the policy. Keep them roughly in [-1, 1] (obs normalization helps but won't fix wild scales).
- `class Scheme`:
  - `__init__(self, params: dict)`
  - `reset(self, env)`: episode start, after the first target spawned.
  - `set_progress(self, p)`: training progress 0..1 (curriculum your reward if you like). Evaluation uses p=1.
  - `extra_obs(self, env) -> list[float]`: current extra features (called by observe(); must be finite, len == EXTRA_OBS).
  - `step(self, ctx) -> float`: shaped reward for this step (called every step, after reach/crash flags are known, before the target respawns).
  - optional `on_target_spawn(self, env)`: called after every target respawn (after a reach), e.g. to re-baseline a potential.

The env separately adds `reach_reward` on every reach and subtracts `crash_penalty` on a crash; do not add those yourself (but you may pay extra on
`ctx.reached` / `ctx.crashed`). Reward scaling in the trainer normalizes magnitudes, but ratios between terms matter.

## `env` (read-only use) in reset/extra_obs/on_target_spawn
`env.target` (np array x,y), `env.drone` (`.shell.center`, `.shell.vel`, `.shell.angle`, `.shell.angular_velocity`, `.thruster_forces`, `.thruster_angles`),
`env.time_to_go()`, `env.dist_to_target()`, `env.steps`, `env.progress`, `env.prev_action`. Everything is in the physical frame.

## `ctx` fields (plain floats; physical frame, so mirroring never changes the reward)
`dist, prev_dist` (px to target now / last step; `prev_dist` is to the same target, re-set on respawn), `dx, dy` (vector to target), `vx, vy, speed` (centre
velocity, px/s), `approach_speed` (velocity component toward target, signed), `lateral_speed` (>=0, perpendicular component), `angle` (body angle rad,
signed), `ang_vel`, `prev_ang_vel`, `forces`/`prev_forces` (list of 2, 0..1), `angles_out` (list of 2, rad; >0 = nozzle points outward), `action`/`prev_action`
(np arrays of 4, physical frame), `steps` (in episode), `progress` (0..1), `reached` (bool, this step), `crashed` (bool), `hold_steps` (calm steps in a row
including this one), `time_to_go` (bang-bang time-optimal estimate, accel 6000 px/s^2; on a reach step it refers to the old target), `W, H` (world size),
`x, y`, `dist_to_wall` (min distance to a border).
Reach rule during training ramps from loose to strict (see spec `reach_speed`, `reach_ang`, `reach_hold`); evaluation always uses the strict rule
(within 20 px, speed < 20, |spin| < 0.5, held 25 steps). Episodes are 4096 steps, 50 steps = 1 s.
