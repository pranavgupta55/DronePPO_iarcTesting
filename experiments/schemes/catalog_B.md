# Catalog B: ideas 11-20

All schemes subclass `time_shaping` (ideas 11-20 each add one term on top). The shared base params are:
- `shaping_w` default 100 (sweep 10 to 2000; 2000 is time_shaping's value, lowered so a cruising step is about +2 and not +40)
- `time_penalty` default 0.5 (sweep 0.05 to 5)
- `ang_vel_pen` default 0.5 (sweep 0.05 to 5), `deadband` default 0.3 (sweep 0.1 to 1 rad/s)

Each row: name | description | default params (sweep range) | EXTRA_OBS

11. crash_cost_scaled_by_remaining_value | On a crash, extra cost `crash_k * clip(time_to_go, t_floor, t_cap)` so early crashes cost more than late ones. | crash_k 1.0 (0.1 to 10); t_floor 1.0 s (0.1 to 5); t_cap 10 s (fixed, 5 to 30) | 0

12. altitude_floor_gravity_barrier | Penalises falling speed only near the floor: `barrier_k * vdown^2 * exp(-h/h0)`, with h = H - y (gravity pulls toward +y, so the floor is y = H; the idea's text labels y as height, but the formula needs h above the floor). | barrier_k 1e-5 (1e-6 to 1e-3); h0 100 px (30 to 300) | 0

13. survival_clearance_bonus | Small per-step bonus for distance from the nearest wall, `clear_c * (1 - anneal*progress) * min(1, wall / (clear_frac*min(W,H)))`, kept below time_penalty. | clear_c 0.1 (0.01 to 0.4); anneal 0.5 (0 to 1); clear_frac 0.25 (0.1 to 0.5) | 0

14. lagrangian_crash_budget | Crash and near-wall penalty weighted by a dual variable `lam`, updated once per episode as `lam += eta * (crashed - budget)`, clipped to [0, lam_max]. Lam is per env worker. | lam0 1.0 (0.1 to 100); eta 0.01 per episode (1e-3 to 1e-1); budget 0.02 (0.005 to 0.1); crash_w 10 (1 to 100); wall_w 0.05 (0.01 to 0.5); lam_max 100 (fixed) | 0

15. speed_gated_highpass_wobble | Penalises the high-passed spin `hp = w - EMA(w)` squared, only while slow: `wobble_k * hp^2 * max(0, 1 - speed/v_low)`. | wobble_k 1.0 (0.01 to 100); lp_alpha 0.1 (0.02 to 0.5); v_low 150 px/s (50 to 500) | 0

16. zero_crossing_rocking_counter | Counts spin sign flips in a rolling window (|w| < w_eps is ignored), and pays `flip_k * flips` per step while speed < v_low (hard gate). Each flip costs about `flip_k * window` in total. | flip_k 0.05 (0.005 to 0.5); window 60 steps (20 to 200); w_eps 0.05 rad/s (0.01 to 0.2); v_low 150 px/s (50 to 500) | 0

17. damping_energy_dissipation | Charges only increases of rotational energy `w^2` between steps, and only while speed < v_low (hard gate). Spin decay is free. | damp_k 5.0 (0.5 to 500); v_low 150 px/s (50 to 500) | 0

18. hover_only_gate_penalty | Penalises `w^2 + angle^2` with two smooth sigmoid gates: slow (speed < v_low) and near the target (dist < d_gate). Fast flight is not affected. Near the target while slow, tilt is penalised by design. | hover_k 0.5 (0.01 to 10); v_low 150 px/s (50 to 500); v_soft 30 px/s (5 to 100); d_gate 100 px (50 to 300); d_soft 20 px (5 to 60) | 0

19. settle_bonus_in_hold | Positive bonus while inside the reach radius and the hold counter runs: `settle_k * min(1, hold/hold_max) * exp(-|w|/w0)`. | settle_k 1.0 (0.1 to 10); hold_max 25 steps (10 to 50); w0 0.5 rad/s (0.05 to 2); radius 20 px (fixed; equals REACH_DIST) | 0

20. action_chatter_at_low_speed | Smoothness penalty on the 4-dim action change, only while slow: `chatter_k * |a - a_prev|^2 * max(0, 1 - speed/v_low)`. | chatter_k 0.02 (1e-3 to 1); v_low 150 px/s (50 to 500) | 0
