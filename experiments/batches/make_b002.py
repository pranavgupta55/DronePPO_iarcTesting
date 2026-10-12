"""Generate batches/b002.json (primary-weight x10/x40/x160 sweep for 30 schemes) and batches/b002_notes.md.

Defaults are read from each scheme module's DEFAULTS, so the values are always the real ones.
Run from the experiments directory:  python3 batches/make_b002.py
"""
import importlib
import json
import os
import sys

sys.dont_write_bytecode = True  # do not write __pycache__ into schemes/

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

MULTIPLIERS = (("x10", 10), ("x40", 40), ("x160", 160))

# scheme -> primary param(s). Rules:
#  - approach_w (catalog A) and shaping_w (catalog B/C, inherited base shaping) are the main progress term.
#  - Extra positive goal-directed bonuses/progress terms are included (save_k, record_k, base_k, entry_k, k, w,
#    settle_k, proj_w). Penalties, barriers, chatter, spin/time terms and nozzle/coverage terms are excluded.
#  - reward_bucket_normalisation: shaping_w is divided out by a running RMS, so w_shape is the real weight.
PRIMARY = {
    # Catalog A
    "switching_curve_margin": ["approach_w"],
    "time_saved_per_target": ["save_k", "approach_w"],
    "lyapunov_time_potential": ["w"],
    "progress_per_effort": ["k"],
    "entry_speed_fly_through": ["entry_k", "approach_w"],
    "self_record_split_bonus": ["record_k", "approach_w"],
    "curriculum_time_baseline": ["base_k", "approach_w"],
    "bang_bang_thrust_bonus": ["approach_w"],
    "time_to_wall_barrier": ["approach_w"],
    "viability_hinge": ["approach_w"],
    # Catalog B
    "crash_cost_scaled_by_remaining_value": ["shaping_w"],
    "altitude_floor_gravity_barrier": ["shaping_w"],
    "survival_clearance_bonus": ["shaping_w"],
    "lagrangian_crash_budget": ["shaping_w"],
    "speed_gated_highpass_wobble": ["shaping_w"],
    "zero_crossing_rocking_counter": ["shaping_w"],
    "damping_energy_dissipation": ["shaping_w"],
    "hover_only_gate_penalty": ["shaping_w"],
    "settle_bonus_in_hold": ["shaping_w", "settle_k"],
    "action_chatter_at_low_speed": ["shaping_w"],
    # Catalog C
    "thrust_asymmetry_at_hover": ["shaping_w"],
    "thrust_weighted_nozzle_alignment": ["shaping_w"],
    "nozzle_alignment_potential": ["shaping_w"],
    "nozzle_annealed_curriculum": ["shaping_w"],
    "thrust_to_target_projection": ["shaping_w", "proj_w"],
    "progress_coverage_bonus": ["shaping_w"],
    "stage_gated_reward_mixture": ["shaping_w"],
    "multiplicative_safety_gate": ["shaping_w"],
    "lexicographic_priority_scalarisation": ["shaping_w"],
    "reward_bucket_normalisation": ["w_shape"],
}

SCHEMES_DIR_NAMES = [
    "switching_curve_margin", "time_saved_per_target", "lyapunov_time_potential", "progress_per_effort",
    "entry_speed_fly_through", "self_record_split_bonus", "curriculum_time_baseline", "bang_bang_thrust_bonus",
    "time_to_wall_barrier", "viability_hinge",
    "crash_cost_scaled_by_remaining_value", "altitude_floor_gravity_barrier", "survival_clearance_bonus",
    "lagrangian_crash_budget", "speed_gated_highpass_wobble", "zero_crossing_rocking_counter",
    "damping_energy_dissipation", "hover_only_gate_penalty", "settle_bonus_in_hold", "action_chatter_at_low_speed",
    "thrust_asymmetry_at_hover", "thrust_weighted_nozzle_alignment", "nozzle_alignment_potential",
    "nozzle_annealed_curriculum", "thrust_to_target_projection", "progress_coverage_bonus",
    "stage_gated_reward_mixture", "multiplicative_safety_gate", "lexicographic_priority_scalarisation",
    "reward_bucket_normalisation",
]


def clean(v):
    return round(float(v), 10)


def main():
    assert sorted(PRIMARY) == sorted(SCHEMES_DIR_NAMES), "scheme list mismatch"
    assert len(PRIMARY) == 30

    specs = []
    notes = []
    for scheme in SCHEMES_DIR_NAMES:
        mod = importlib.import_module(f"schemes.{scheme}")
        defaults = mod.DEFAULTS
        primary = PRIMARY[scheme]
        for p in primary:
            assert p in defaults, f"{scheme}: primary {p} not in DEFAULTS"
        notes.append(
            f"{scheme}: primary = " + ", ".join(f"{p} (default {defaults[p]})" for p in primary)
        )
        for suffix, mult in MULTIPLIERS:
            params = {p: clean(defaults[p] * mult) for p in primary}
            specs.append({"name": f"{scheme}_{suffix}", "scheme": scheme, "params": params, "seed": 1})

    batch = {"batch": "b002", "specs": specs}
    with open(os.path.join(HERE, "b002.json"), "w") as f:
        json.dump(batch, f, indent=2)
        f.write("\n")
    with open(os.path.join(HERE, "b002_notes.md"), "w") as f:
        f.write("\n".join(notes) + "\n")

    # validation
    with open(os.path.join(HERE, "b002.json")) as f:
        loaded = json.load(f)
    assert len(loaded["specs"]) == 90, len(loaded["specs"])
    names = [s["name"] for s in loaded["specs"]]
    assert len(set(names)) == len(names), "duplicate names"
    for s in loaded["specs"]:
        d = importlib.import_module(f"schemes.{s['scheme']}").DEFAULTS
        for k in s["params"]:
            assert k in d, f"{s['name']}: {k} not in DEFAULTS"
    print(f"ok {len(loaded['specs'])} specs")


if __name__ == "__main__":
    main()
