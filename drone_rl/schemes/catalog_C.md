# Scheme catalog C (ideas 21-30)

Unless marked "no base", each scheme extends `time_shaping` (time penalty, time-to-go potential, spin penalty above a deadband), with the base weights
below. Base params are scaled to keep per-step magnitudes near O(0.1-10): `shaping_w` is 50 here (time_shaping uses 2000).

Base sweep: `time_penalty` 0.5 (0.05-5), `shaping_w` 50 (5-500), `ang_vel_pen` 0.5 (0.05-5), `deadband` 0.3 (0.1-1).

Format: name | description | default params (sweep range) | EXTRA_OBS

thrust_asymmetry_at_hover | penalises |F1-F2| while hovering (speed and distance ramps to 0 at v_low, d_low) | asym_w 0.5 (0.01-5); v_low 30 px/s (10-100); d_low 100 px (50-200) | 0

thrust_weighted_nozzle_alignment | rewards sum_i F_i cos(outward_i - target), so alignment only counts when thrust is on | align_w 0.5 (0.01-5); outward_target pi/6 rad (fixed) | 0

nozzle_alignment_potential | PBRS term on nozzle distance to the outward target, Phi = -sum_i abs(outward_i - target) (rad), with gamma 0.99 | nozzle_w 2.0 (0.2-20); gamma 0.99 (0.9-0.999) | 0

nozzle_annealed_curriculum | thrust-weighted nozzle alignment whose weight falls linearly to 0 at progress zero_at | nozzle_w 0.5 (0.05-5); zero_at 0.7 (0.3-0.9) | 0

thrust_to_target_projection | rewards thrust pushing toward the target: sum_i F_i cos(psi_i), psi_i between thruster push direction and the vector to target | proj_w 0.5 (0.01-5) | 0

progress_coverage_bonus | pays cov_w/sqrt(visits+1) per entry into an 8x6 grid cell above v_thresh; re-entries also pay -revisit_pen*cov_w (per episode) | cov_w 1.0 (0.1-5); v_thresh 60 px/s (10-150); revisit_pen 0.5 (0-1) | 0

stage_gated_reward_mixture | w_speed(p) * base + safe_w * wall-proximity penalty + w_nozzle(p) * nozzle_w * alignment; w_speed ramps up, w_nozzle ramps down | speed_start 0.3 (0-1); speed_ramp_end 0.3 (0.1-1); safe_w 1.0 (0.1-10); wall_ref 60 px (20-200); nozzle_w 0.5 (0.05-5); nozzle_end 0.7 (0.3-1) | 0

multiplicative_safety_gate | base shaping term multiplied by sigmoid((wall - wall_mid)/wall_scale); time and spin penalties not gated (no base for the gate, standalone) | wall_mid 40 px (10-100); wall_scale 10 px (2-40) | 0

lexicographic_priority_scalarisation | r = base r_speed if r_safe > -safe_eps, else r_safe (wall-proximity penalty); no weight trade-off (standalone) | wall_ref 60 px (20-200); wall_w 1.0 (fixed); safe_eps 0.05 (0.01-1) | 0

reward_bucket_normalisation | sum of four terms (shaping, time, spin, wall), each divided by its running RMS (EMA 1e-3), clipped to [-1,1], times weight (standalone) | w_shape 1.0 (0.1-1); w_time 0.2 (0.1-1); w_spin 0.3 (0.1-1); w_wall 0.3 (0.1-1); ema 1e-3 (1e-4-1e-2); floor 1e-2 (1e-4-1) | 0
