# Reward-design ideas for the 2-thruster PPO drone

## Literature notes (quick check, 1-2 sources each)

- **Champion-level drone racing (Swift, Kaufmann et al., Nature 2023)**: model-free on-policy RL in simulation, then transfer to hardware. Reward details were not visible in the abstract. Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10468397/
- **Structured RL for time-optimal quadrotor flight (Barhoum & Pyrkin, 2025)**: composite reward on radial velocity and acceleration to push toward bang-bang-like manoeuvres. Closest match to the goal. Source: https://old-pribor.itmo.ru/en/article/24003/strukturirovannoe_obuchenie_s_podkrepleniem_dlya_optimalnogo_po_vremeni_poleta_kvadrokoptera.htm
- **Time-optimal flight in cluttered environments via safe RL (arXiv 2406.19646)**: sparse minimum-time reward is hard to learn, so a gate-progress proxy plus an angular-rate penalty and a terminal early-finish reward are used. Source: https://arxiv.org/pdf/2406.19646
- **Simultaneous state-to-state minimum-time planning and control (arXiv 2510.20008)**: proxy reward from point-mass trajectories; without acceleration and angular-rate penalties the drone flew fast but failed to hover stably. Source: https://arxiv.org/pdf/2510.20008
- **Potential-based shaping (Ng, Harada & Russell 1999)**: shaping term F = gamma*Phi(s') - Phi(s) leaves the optimal policy unchanged. Secondary sources: https://arxiv.org/pdf/2208.09570
- **Reward scale balance (How to Train your Quadrotor, arXiv 2012.06656)**: penalty terms differing by orders of magnitude interfere; clipping each penalty into a bounded [0,1] form helps. Source: https://arxiv.org/pdf/2012.06656
- **Barrier-based reward shaping (review, arXiv 2508.09128)**: soft barrier penalties are easy to add to actor-critic training but give weaker guarantees; badly tuned weights either miss violations or make the policy too timid. Source: https://arxiv.org/pdf/2508.09128

Terms used below: **PBRS** = potential-based reward shaping; **CBF** = control barrier function; **Lagrangian** = a constraint whose penalty weight is adapted automatically during training.

## Notation

Quantities as given: `d` distance to target, `d_prev`, `va` approach speed, `vlat` lateral speed, `v` speed, `th` body angle, `w` angular velocity, `w_prev`, `F1, F2` thrust (0..1), `phi1, phi2` nozzle outward angles, `a` action, `a_prev`, `p` training progress (0..1), `hold` steps held near target, `tg` time-to-go (bang-bang minimum time), `wall` distance to nearest wall, `W, H` world size. Derived: `amax = 6000` px/s^2, `dt` step length, `brake = va^2 / (2*amax)` (braking distance, extra feature).

---

## Goal 1: speed (reach targets quickly)

**1. switching_curve_margin** (extra feature: `sigma = d - brake`)
- Formula: `r = -k * max(0, -sigma) * (va > 0)`; penalise only when the drone is closing faster than it can brake.
- Why: teaches braking exactly at the switching curve, which is the bang-bang rule. Dense and physically grounded.
- Failure: the drone may brake early and creep (sigma stays slightly positive and it stops far from target). Mitigate with a small `+k2*va` term.
- Weight: 0.01 to 3 per px.

**2. time_saved_per_target** (event-based, sparse)
- Formula: on reach, `r = k * (T_ref - t_elapsed)`, where `T_ref = tg_at_spawn * 1.3`.
- Why: pays for time actually saved against an estimate of the optimum, not a flat bonus. Replaces the one-off 10000 with a relative quantity.
- Failure: if `T_ref` is loose, there is no pressure; if tight, reach rate collapses early. Anneal `T_ref` factor with `p` from 3.0 to 1.1.
- Weight: 5 to 500 per step saved.

**3. lyapunov_time_potential** (PBRS, Lyapunov-style)
- Formula: `V = tg + c*va` ; `r = -(V_t - V_{t-1})` on the same step scale.
- Why: a Lyapunov-like decreasing function of the minimum-time estimate. Adding the approach-speed term rewards the state, not only the time.
- Failure: `V` can fall by letting `va` grow, then crash into a wall. Pair with idea 14.
- Weight: 0.05 to 2.

**4. progress_per_effort** (efficiency ratio)
- Formula: `r = (d_prev - d) / (dt * (eps + F1 + F2))`, clipped to [-5, 5].
- Why: rewards getting closer per unit of thrust. Encourages bang-bang (full thrust, then full brake) and discourages wasted thrust while cruising.
- Failure: near zero thrust the ratio blows up; the clip and eps are essential.
- Weight: 0.01 to 1.

**5. entry_speed_fly_through** (event-based, two-stage)
- Formula: on first entry to radius 20, `r = k * va_entry`; then a separate stop term applies within the hold window.
- Why: makes the agent aim to arrive fast and then stop, not to crawl in. Splits "fast" and "stop" into two events.
- Failure: the agent overshoots, arrives fast and fails to stop, then loses the hold. Keep the stop penalty for leaving the radius before hold completes.
- Weight: 0.5 to 20 per entry.

**6. self_record_split_bonus** (self-competition, internal state)
- Formula: store `best_split[k]` for target index k across episodes. On reach, `r = k * max(0, best_split[k] - split_t)`.
- Why: a moving benchmark that scales with the agent's own improvement; avoids a hand-set reference.
- Failure: noisy early records; the agent may repeat easy targets. Use EMA of the best with a floor.
- Weight: 1 to 100 per step beaten.

**7. curriculum_time_baseline** (advantage-like, internal state)
- Formula: keep an EMA `B[bucket]` of reach time per distance bucket; `r = k * (B - t_elapsed) / B` on reach.
- Why: pays only for beating the running average at that difficulty. Gives a fair signal across distances.
- Failure: the baseline lags if the policy improves fast, so rewards shrink (use fast EMA).
- Weight: 1 to 50.

**8. bang_bang_thrust_bonus** (direct, action-shaped)
- Formula: `r = k * (|F1 - 0.5| + |F2 - 0.5|) * 2 * (speed > v_thresh)`.
- Why: pushes thrust to the extremes, as bang-bang control does, only while moving.
- Failure: the agent saturates thrusters into oscillation. Gate with `a_prev` sign changes or idea 17.
- Weight: 0.001 to 0.1.

---

## Goal 2: crash avoidance (stay in the world)

**9. time_to_wall_barrier** (CBF-style, time-based)
- Formula: `tw = wall / max(eps, v_wall)` where `v_wall` is velocity toward the nearest wall; `r = -k * exp(-tw / tau)`, `tau = 0.5 s`.
- Why: penalises based on time left before impact, which is independent of speed. A fast drone far from a wall is safe; a slow drone very close is not.
- Failure: the drone may avoid walls by hovering. Gate with idea 14.
- Weight: 0.1 to 10.

**10. viability_hinge** (CBF-style, braking feasibility)
- Formula: `m = wall - v_wall^2 / (2*amax)`; `r = -k * max(0, -m)`.
- Why: penalises only states from which the drone can no longer stop before the wall. Direct version of the "viability kernel" idea.
- Failure: the bound `amax` is optimistic for the real thruster limits; use a lower value (e.g. 4000 px/s^2) to be safe.
- Weight: 0.5 to 20.

**11. crash_cost_scaled_by_remaining_value** (event-based, value-aware)
- Formula: on crash, `r = -k * tg` (time-to-go at crash), instead of a flat penalty.
- Why: an early crash, with much of the episode still to earn, costs more than a late one. Matches expected lost reward.
- Failure: late-episode crashes become cheap, so the agent may take risks near the end. Add a floor: `max(tg, t_floor)`.
- Weight: 0.5 to 5 per second of remaining tg.

**12. altitude_floor_gravity_barrier** (gravity-aware)
- Formula: `r = -k * max(0, vy_down)^2 * exp(-(H - y)/h0)` where `y` is height above bottom.
- Why: gravity makes the floor the main danger. Penalises downward speed only near the floor.
- Failure: ignores side walls, so the drone may move sideways into a wall. Use a per-wall version.
- Weight: 1e-4 to 1e-2 (scale to the squared speed).

**13. survival_clearance_bonus** (dense, margin-based)
- Formula: `r = c * min(1, wall / (0.25 * min(W,H)))` per step.
- Why: rewards clearance from walls, not just survival. Tilting is not penalised.
- Failure: hovering near the centre of the world maximises it. Keep weight small (below one-off reach reward) and cap with `p`.
- Weight: 0.001 to 0.05.

**14. lagrangian_crash_budget** (constrained RL, adaptive)
- Formula: `r = r_speed - lambda * [crashed or wall < 20]`, with `lambda` updated each batch: `lambda += eta * (crash_rate - budget)`, `budget = 0.02`.
- Why: lets the agent go fast while keeping crashes under a set rate. Removes hand-tuning of the crash weight.
- Failure: `lambda` oscillates early in training; use a warm-up and slow `eta`.
- Weight: `eta` 1e-3 to 1e-1; initial `lambda` 1 to 100.

---

## Goal 3: anti-wobble / anti-tilt at hover (fast flight not penalised)

**15. speed_gated_highpass_wobble** (frequency-based)
- Formula: `lp = 0.9*lp + 0.1*w`; `hp = w - lp`; `r = -k * hp^2 * max(0, 1 - v/v_low)`.
- Why: penalises high-frequency angular content only when slow. Fast flight, where rotation is intentional, is not affected.
- Failure: the low-pass filter lags, so smooth but slow tilt changes count as wobble. Tune the EMA.
- Weight: 1e-3 to 1 (scale to w^2).

**16. zero_crossing_rocking_counter** (event-based, internal state)
- Formula: count sign flips of `w` in a rolling window of 60 steps; `r = -k * flips * (v < v_low)`.
- Why: rocking shows as frequent sign flips. Counting them is less sensitive to amplitude than a squared penalty.
- Failure: counts flips during the final approach at low speed, which may be a legitimate braking manoeuvre. Keep the speed gate tight.
- Weight: 0.01 to 1 per flip.

**17. damping_energy_dissipation** (energy-based)
- Formula: `E_rot = w^2`; `r = -k * max(0, E_rot - E_rot_prev) * (v < v_low)`.
- Why: penalises only increases of rotational energy while nearly stopped. Decay is free, so the agent may damp freely.
- Failure: the agent may spin up slowly, which this misses. Combine with idea 15.
- Weight: 0.01 to 2.

**18. hover_only_gate_penalty** (gated direct penalty)
- Formula: `r = -k * (w^2 + th^2) * [v < v_low and d < 100]`.
- Why: the penalty is zero everywhere except near-hover states. Direct response to the requirement "no hover tilt", without touching flight.
- Failure: a hard gate creates a reward cliff; the agent may learn to pass the gate quickly. Use a smooth sigmoid gate.
- Weight: 1e-3 to 0.5.

**19. settle_bonus_in_hold** (sparse-in-time, positive)
- Formula: while inside radius, `r = k * (hold / hold_max) * exp(-|w|/w0)`.
- Why: rewards a calm settle only during the hold window. Its magnitude grows as the drone holds, so the drone learns to settle, not hover.
- Failure: it may rely on the zone reward that was already tried; keep the zone term out.
- Weight: 0.5 to 10 per step.

**20. action_chatter_at_low_speed** (gated smoothness)
- Formula: `r = -k * |a - a_prev|^2 * max(0, 1 - v/v_low)`.
- Why: applies the smoothness penalty only when slow, so bang-bang at speed is not penalised. Differs from the ungated smoothness tried earlier.
- Failure: the agent may save chatter for the fast phase. Accept this; it is what the user wants.
- Weight: 1e-3 to 0.1.

**21. thrust_asymmetry_at_hover** (direct, hover-specific)
- Formula: `r = -k * |F1 - F2| * [v < v_low and d < 100]`.
- Why: a hovering drone with unequal thrust is rotating or tilting. Penalises that state only.
- Failure: the drone may balance thrust while still tilting from nozzle swivel. Combine with idea 18.
- Weight: 1e-3 to 0.1.

---

## Goal 4: outward nozzles

**22. thrust_weighted_nozzle_alignment**
- Formula: `r = k * sum_i F_i * cos(phi_i - 30 deg)`.
- Why: alignment matters only when thrust is applied; a nozzle at the wrong angle with no thrust costs nothing. Gives the outward-nozzle objective a physical meaning.
- Failure: the agent keeps thrust low on the misaligned side and loses control authority. Check the left/right balance.
- Weight: 0.001 to 0.1.

**23. nozzle_alignment_potential** (PBRS)
- Formula: `Phi = -(|phi1 - 30| + |phi2 - 30|)`; `r = gamma*Phi(s') - Phi(s)`.
- Why: a potential-based term does not change the optimal policy in the PBRS sense; only the learning path changes.
- Failure: the optimal policy still ignores the nozzle, because the potential does not require thrust. Combine with idea 22 for a real incentive.
- Weight: 0.001 to 0.05.

**24. nozzle_annealed_curriculum**
- Formula: `r = k * (1 - p) * alignment` (alignment as in idea 22 or 23); weight ramps to 0 by `p = 0.7`.
- Why: enforces the nozzle rule early, when the policy is learning basic control, and removes it later so speed is free.
- Failure: the agent may learn to ignore nozzles once the weight reaches zero; check the nozzle angles near the end of training.
- Weight: 0.01 to 0.5 at start.

**25. thrust_to_target_projection** (direct, useful-thrust)
- Formula: `r = k * (F1*cos(psi1) + F2*cos(psi2))`, where `psi_i` is the angle between the nozzle exhaust direction and the vector to target (`vec`).
- Why: rewards thrust that points the drone toward the target; the nozzle angle then matters only for the useful direction.
- Failure: the exhaust may be pointed toward the target while the drone rotates; watch the angle term.
- Weight: 0.001 to 0.05.

---

## Combined and meta systems

**26. progress_coverage_bonus** (exploration, "use the whole world")
- Formula: grid the world into 8x6 cells; `r = k / sqrt(visits[cell] + 1)` when the drone enters an unvisited or rarely visited cell at speed `v > v_thresh`.
- Why: encourages visiting the whole world, which is needed for targets anywhere, and only while moving.
- Failure: the agent flies in circles to hit new cells. Penalise cells visited more than once per episode.
- Weight: 0.1 to 5 per new cell.

**27. stage_gated_reward_mixture** (curriculum via `p`)
- Formula: `r = w_speed(p) * r_speed + w_safe(p) * r_safe + w_nozzle(p) * r_nozzle`, with `w_speed` ramping up and `w_nozzle` ramping down.
- Why: sequences objectives so each is learnt in its own phase. Simple and reproducible.
- Failure: the drop of `w_nozzle` may cause nozzle drift.
- Weight: multipliers in [0, 1].

**28. multiplicative_safety_gate**
- Formula: `r = r_speed * sigmoid((wall - 40) / 10) * sigmoid((hold_margin - 0) / 5)`.
- Why: speed reward is only active when safe. A crash is penalised by losing all speed reward, without a large negative value.
- Failure: the agent may stop at walls to keep reward. Keep a small crash penalty too.
- Weight: multiplicative; no separate weight.

**29. lexicographic_priority_scalarisation**
- Formula: compute `r_safe` (crash, wall barrier) and `r_speed` separately; use `r = r_speed` only if `r_safe > -eps`, otherwise `r = r_safe`.
- Why: enforces the priority order (1) speed, (2) crash, (3) wobble strictly, instead of weighting them against each other. Avoids weight tuning.
- Failure: PPO's advantage normalisation may distort the lexicographic switch; test on a small batch.
- Weight: no weights; threshold `eps` 0.01 to 1.

**30. reward_bucket_normalisation** (scale-invariant)
- Formula: each term `r_j` is divided by its running standard deviation, then clipped to [-1, 1] and summed.
- Why: fixes scale problems across terms (quadrotor paper). Lets weights be near 1 without scale tuning.
- Failure: the normalisation hides true magnitudes; the bang-bang term may look too small.
- Weight: all terms in [0.1, 1] after normalisation.

---

## Try-first shortlist (ranked)

1. **switching_curve_margin (1)**: dense, bang-bang-aligned braking rule; the most direct fix for speed.
2. **time_to_wall_barrier (9)**: speed-independent wall barrier; avoids crashes without penalising fast flight.
3. **speed_gated_highpass_wobble (15)**: targets the wobble requirement without touching fast flight.
4. **time_saved_per_target (2)**: replaces the flat 10000 reach bonus with a relative time-saved bonus.
5. **crash_cost_scaled_by_remaining_value (11)**: makes crashes costly in proportion to the time they waste.
6. **thrust_weighted_nozzle_alignment (22)**: makes the nozzle rule matter only when thrust is applied.
7. **hover_only_gate_penalty (18)**: a tilt penalty that only applies in the hover state.
8. **lagrangian_crash_budget (14)**: removes hand-tuning of the crash weight.
9. **progress_per_effort (4)**: a clean efficiency ratio that encourages full-thrust bursts.
10. **lexicographic_priority_scalarisation (29)**: enforces the priority order directly, avoiding weight search.
