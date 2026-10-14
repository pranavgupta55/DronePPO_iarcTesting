"""Hand-written bang-bang controller for the ceiling check: python3 bangbang_controller.py  (sweeps a few settings with the fixed evaluation rules).
The drone can rotate its whole body, so the thrust can point in any direction: it turns to point its thrust along the line to the target, thrusts hard,
turns 180 degrees at the switching point (remaining distance = stopping distance) and thrusts hard the other way to stop on the target.
Measured with the defaults below in the 2400x1600 world (24-36 episodes x 2048 steps, evaluation rules): ~13.8 reaches/episode, ~3% crashes, ~377 px/s (at 1x thrust, a_acc 1500 / a_brake 900: 11.9 reaches, 0% crashes, 310 px/s; at 3x: 14.1 reaches, 8% crashes; 5x: 25% crashes).
The braking latches (no accelerate-brake-accelerate), the body turns back the way it came instead of flipping a full 360 degrees, and a last-resort wall brake pushes away from a nearby wall. Sideways drift is cancelled by the same thrust vector. The commanded acceleration -> a thrust vector (cancelling gravity) -> a body angle and a force;
a rotation loop (fastest rotation that can still stop in time) turns the body with the force difference between the two thrusters."""
import itertools
import math
import multiprocessing as mp
import random

import numpy as np

from drone_rl.config import GRAVITY, MAX_THRUSTER_FORCE, DRONE_SIZE, THRUSTER_NEUTRAL_ANGLE, THRUSTER_INWARD_ALLOWANCE

DEFAULTS = {"a_acc": 3000.0, "a_brake": 1800.0, "margin": 1.7, "outward_deg": 0.0, "near": 80.0, "t_min": 150.0, "wall_margin": 1.0, "t_lag": 0.08, "w_turn": 12.0, "kp_pos": 14.0, "kd_pos": 8.0, "k_lat": 6.0,
            "f_max": 1.0, "f_rot": 0.08, "gate_power": 3.0, "alpha_lim": 300.0, "w_max": 14.0, "vector_deg": 0.0, "wall_brake": 1.3, "wall_pad": 10.0, "wall_trigger": 0.3, "latch_near": 0, "wind_limit": 1.5, "release_speed": 25.0, "release_ratio": 3.0, "release_px": 150.0, "brake_gain": 1.15, "k_lin": 15.0, "k_w": 40.0}


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def control(env, p):
    d, sh = env.drone, env.drone.shell
    pos = np.array([float(sh.center[0]), float(sh.center[1])])
    vel = np.array([(float(sh.vel[0, 0]) + float(sh.vel[1, 0])) / 2, (float(sh.vel[0, 1]) + float(sh.vel[1, 1])) / 2])
    to_target = env.target - pos
    dist = float(np.hypot(*to_target))
    u = to_target / (dist + 1e-9)
    along = float(vel @ u)
    lateral = vel - along * u
    if dist > p["near"] or (p["latch_near"] and getattr(env, "bb_braking", False)):  # a started brake carries on into the final approach until nearly stopped
        brake_thrust = (-p["a_brake"] * u[0], -p["a_brake"] * u[1] - GRAVITY)  # the thrust direction the braking needs, and how far the body still has to turn to it
        turn_time = abs(wrap(math.atan2(brake_thrust[0], -brake_thrust[1]) - sh.angle)) / p["w_turn"]
        braking_distance = p["margin"] * along * abs(along) / (2 * p["a_brake"]) + (p["t_lag"] + turn_time) * along  # plus the distance covered while turning around
        braking = getattr(env, "bb_braking", False)  # latched: once braking starts it continues until the drone is nearly stopped (re-deciding every step made it brake, speed up again, brake again)
        if braking and (along < p["release_speed"] or dist > braking_distance * p["release_ratio"] + p["release_px"]):
            braking = False  # stopped, or the target jumped far away
        elif not braking and dist <= braking_distance and along > 0:
            braking = True
        env.bb_braking = braking
        if braking:  # brake only as hard as needed to stop exactly on the target (full a_brake at most), so one deceleration is enough
            brake = min(p["a_brake"], p["brake_gain"] * along * along / (2 * max(dist - p["t_lag"] * along * 0.5, 5.0)))
            accel = -brake * u - p["k_lat"] * lateral
        else:
            accel = p["a_acc"] * u - p["k_lat"] * lateral
    else:  # final approach: a soft PD onto the target so it settles inside the reach rule
        accel = p["kp_pos"] * to_target - p["kd_pos"] * vel
    for axis, size in ((0, env.W), (1, env.H)):  # wall awareness: too fast toward a wall to stop in time -> stop accelerating toward it
        speed_axis = vel[axis]
        wall_dist = (size - pos[axis]) if speed_axis > 0 else pos[axis]
        if abs(speed_axis) > 1.0 and p["wall_margin"] * speed_axis * speed_axis / (2 * p["a_brake"]) > max(wall_dist, 1.0):
            if accel[axis] * speed_axis > 0:  # only stop pushing toward the wall; the target logic does the braking (forced braking made it chatter)
                accel[axis] = 0.0
        if p["wall_brake"] and abs(speed_axis) > 5.0:  # last resort near a wall: push away from it just hard enough to stop before it
            need = p["wall_brake"] * speed_axis * speed_axis / (2 * max(wall_dist - p["wall_pad"], 3.0))
            if need > p["wall_trigger"] * p["a_brake"] and -math.copysign(1.0, speed_axis) * accel[axis] < min(need, p["a_brake"]):
                accel[axis] = -math.copysign(min(need, p["a_brake"]), speed_axis)
    thrust = np.array([accel[0], accel[1] - GRAVITY])  # the thrust vector that gives that acceleration (gravity pulls toward +y)
    cos_out = math.cos(math.radians(p["outward_deg"]))  # outward nozzles lose cos(angle) of their push, and of their torque
    force_cmd = min(p["f_max"], float(np.hypot(*thrust)) / (MAX_THRUSTER_FORCE * cos_out))
    # thrust direction is (sin angle, -cos angle); any angle is allowed, including pointing down. A tiny thrust has no meaningful direction: stay upright
    angle_des = math.atan2(thrust[0], -thrust[1]) if np.hypot(*thrust) > p["t_min"] else 0.0
    prev = getattr(env, "bb_unwound", wrap(sh.angle))
    unwound = prev + wrap(sh.angle - prev)  # the body angle tracked continuously (it counts whole turns, so winding up is visible)
    env.bb_unwound = unwound
    candidates = [c for c in (angle_des - 2 * math.pi, angle_des, angle_des + 2 * math.pi) if abs(c) <= math.pi + p["wind_limit"]]
    err = min(candidates, key=lambda c: abs(c - unwound)) - unwound  # turn back the way it came instead of completing a full 360 degree flip; at most wind_limit past half a turn
    # thrust vectoring (vector_deg > 0): tilt both nozzles the same way to swing the thrust up to vector_deg off the body axis without turning the body; the body only turns the rest of the way
    gamma = math.radians(p["vector_deg"])
    tilt = max(-gamma, min(gamma, err)) if gamma > 0 else 0.0
    err_rot = err - tilt
    # rotation loop: turn at the fastest rate from which the body can still stop on the target angle, linear near it
    err_full, err = err, err_rot
    w_des = math.copysign(min(p["w_max"], math.sqrt(p["alpha_lim"] * abs(err)), p["k_lin"] * abs(err)), err)
    angular_accel = p["k_w"] * (w_des - sh.angular_velocity)
    diff = -angular_accel * DRONE_SIZE / (MAX_THRUSTER_FORCE * cos_out)  # f_right - f_left that gives that angular acceleration
    force_mean = max(p["f_rot"] * min(1.0, abs(err) * 2), force_cmd * max(0.0, math.cos(err)) ** p["gate_power"])  # push only when roughly aligned
    force_mean = min(max(force_mean, abs(diff) / 2), 1.0 - abs(diff) / 2)  # leave room for the force difference
    out = math.radians(p["outward_deg"])
    if gamma > 0 and abs(tilt) > 1e-4:
        # both nozzles tilt toward the thrust side; equal push across the stick keeps the torque zero. tan(thrust tilt) = (tan t_in + tan t_out) / 2, the nozzle that
        # swivels outward has the long range (90 deg), the one that swivels inward only 30 deg, so the outward one takes the rest
        t_in = min(THRUSTER_INWARD_ALLOWANCE, math.atan(abs(math.tan(tilt))))
        t_out = min(math.radians(85.0), math.atan(max(0.0, 2 * abs(math.tan(tilt)) - math.tan(t_in))))
        effective = math.atan((math.tan(t_in) + math.tan(t_out)) / 2)
        perp = force_mean * math.cos(effective)
        f_left, f_right = (perp - diff / 2), (perp + diff / 2)
        if tilt > 0:  # thrust leans right: the left nozzle swivels outward, the right one inward
            angle_left, angle_right, cos_left, cos_right = THRUSTER_NEUTRAL_ANGLE + t_out, THRUSTER_NEUTRAL_ANGLE + t_in, math.cos(t_out), math.cos(t_in)
        else:
            angle_left, angle_right, cos_left, cos_right = THRUSTER_NEUTRAL_ANGLE - t_in, THRUSTER_NEUTRAL_ANGLE - t_out, math.cos(t_in), math.cos(t_out)
        force_left, force_right = max(0.0, f_left / cos_left), max(0.0, f_right / cos_right)
        scale = min(1.0, 1.0 / max(force_left, force_right, 1e-9))  # a tilted nozzle needs more than its share of force; scale both down together so clipping one side can't leave a torque
        targets = [force_left * scale, force_right * scale, angle_left, angle_right]
    else:
        targets = [min(1.0, max(0.0, force_mean - diff / 2)), min(1.0, max(0.0, force_mean + diff / 2)), THRUSTER_NEUTRAL_ANGLE + out, THRUSTER_NEUTRAL_ANGLE - out]
    cur = d.thruster_forces + d.thruster_angles
    rate = [d.thruster_force_delta_rate] * 2 + [d.thruster_angle_delta_rate] * 2
    return np.array([(t - c) / r for t, c, r in zip(targets, cur, rate)])


def run(args):
    from drone_rl.env import DroneEnv, make_spec, load_scheme
    p, seed, episodes, steps = args
    random.seed(seed); np.random.seed(seed)
    spec = make_spec({"scheme": "time_shaping", "params": {}})
    reaches, crashes, speeds, peak, mean_force = 0, 0, [], 0.0, []
    for _ in range(episodes):
        module, scheme = load_scheme(spec["scheme"], spec["params"])
        env = DroneEnv(spec, module, scheme, max_steps=steps)
        env.set_ramp(1.0); env.set_eval_mode(); env.reset()
        for t in range(steps):
            _, _, term, trunc = env.step(control(env, p))
            speeds.append(env.drone.shell.linear_velocity_magnitude); peak = max(peak, max(env.drone.thruster_forces)); mean_force.append(max(env.drone.thruster_forces))
            if term: crashes += 1
            if term or trunc: break
        reaches += env.reaches
    return reaches / episodes, crashes / episodes, float(np.mean(speeds)), peak, float(np.mean(mean_force))


if __name__ == "__main__":
    grid = {"a_brake": [900.0, 1500.0], "wall_margin": [1.0, 1.5, 2.5], "outward_deg": [0.0, 30.0]}
    combos = [dict(DEFAULTS, **dict(zip(grid, vals))) for vals in itertools.product(*grid.values())]
    with mp.get_context("spawn").Pool(4) as pool:
        results = pool.map(run, [(c, 11, 10, 2048) for c in combos])
    print("a_brake wall_margin outward | reaches/ep  crash   speed  peak_force  mean_max_force")
    for c, (r, cr, sp, pk, mf) in zip(combos, results):
        print(f"{c['a_brake']:7.0f} {c['wall_margin']:11.1f} {c['outward_deg']:7.0f} | {r:9.1f} {cr:7.0%} {sp:7.0f} {pk:8.2f} {mf:12.2f}")
