# Catalog A (ideas 1-10)

Common dense terms unless noted: `approach_w` * clip(approach_speed / vel_ref, -1, 1), `time_penalty` per step, `ang_vel_pen` * max(0, |spin| - deadband). Defaults: approach_w 1.0, vel_ref 150 px/s, time_penalty 0.1, ang_vel_pen 0.5, deadband 0.3 rad/s.

name | description | params (default; sweep) | EXTRA_OBS
--- | --- | --- | ---
switching_curve_margin | Penalises closing faster than bang-bang braking allows (sigma = d - va^2/(2*amax) < 0), with the common dense terms. | amax 6000 (3000-8000); viol_w 0.01 (0.001-0.3); approach_w 1.0 (0.1-3); sigma_ref 300 px (100-1000) | 1
time_saved_per_target | Pays save_k per second saved against T_ref = time_to_go at spawn * factor, factor annealed 3.0 -> 1.1 with progress. | save_k 10 per s (1-200); factor_start 3.0 (2-5); factor_end 1.1 (1.0-1.5); approach_w 1.0 (0.1-3) | 0
lyapunov_time_potential | Shaping -w * dV/dt (rate clipped) on V = time_to_go + c_va * approach_speed, V = 0 on reach. | w 1.0 (0.05-5); c_va 0.001 s per px/s (1e-4 to 1e-2); rate_clip 5 (1-20) | 0
progress_per_effort | Distance closed per unit of thrust, k * clip((d_prev - d) / (dt * (eps + F1 + F2) * speed_ref), -5, 5). | k 0.1 (0.01-1); eps 0.5 (0.05-2); speed_ref 100 px/s (30-300) | 0
entry_speed_fly_through | Once per target, pays entry_k for approach speed at first entry inside radius; stop penalty inside; leave penalty if exiting before hold_need steps. | entry_k 5 (0.5-20); stop_k 0.5 (0.05-5); leave_k 5 (0.5-20); hold_need 25 steps (10-50); radius 20 px; approach_w 0.5 (0.1-3) | 0
self_record_split_bonus | Pays record_k per second beaten against a cross-episode best split time per target index, with EMA drift up and a floor. | record_k 10 (1-100); ema 0.05 (0.01-0.5); floor_s 0.2 (0.05-1); approach_w 1.0 (0.1-3) | 0
curriculum_time_baseline | Pays base_k * clip((B - t) / B, -1, 1) on reach, with B an EMA of reach time per spawn-distance bucket. | base_k 10 (1-50); ema 0.2 (0.02-0.5); bucket_px 150 (50-300); approach_w 1.0 (0.1-3) | 0
bang_bang_thrust_bonus | Bonus bb_k * (|F1 - 0.5| + |F2 - 0.5|) * 2 while speed > v_thresh. | bb_k 0.01 (0.001-0.1); v_thresh 100 px/s (30-300); approach_w 1.0 (0.1-3) | 0
time_to_wall_barrier | Penalises -k * exp(-tw / tau), tw = smallest distance-to-wall / closing speed over the four walls. | k 1.0 (0.1-10); tau 0.5 s (0.1-2); eps 1 px/s (0.1-10); approach_w 1.0 (0.1-3) | 0
viability_hinge | Penalises -k * violation of the braking margin min_i(dist_i - max(v_i,0)^2/(2*amax)), scaled by 100 px and capped at 10. | k 1.0 (0.5-20); amax 4000 px/s^2 (2000-6000); scale_px 100 (30-300); approach_w 1.0 (0.1-3) | 0
