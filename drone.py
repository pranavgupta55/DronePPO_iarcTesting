import random
import numpy as np
import math

from calcs import clamp, normalize_angle_signed
from config import *


class stick:
    def __init__(self, size):
        self.size = size
        self.points = np.zeros((2, 2), dtype=np.float32)
        self.vel = np.zeros((2, 2), dtype=np.float32)
        self.center = np.zeros(2, dtype=np.float32)
        self.angle = 0.0
        self.previous_angle = 0.0
        self.angular_velocity = 0.0
        self.linear_velocity_magnitude = 0.0
        self.previous_linear_velocity_magnitude = 0.0
        self.drag = 0.0

    def set_physical_state(self, pos, angle, linear_vel_avg=np.array([0., 0.]), angular_vel=0.0):
        self.center = np.array(pos, dtype=np.float32)
        self.angle = normalize_angle_signed(angle)
        self.previous_angle = normalize_angle_signed(angle - angular_vel * TIME_STEP_DELTA)
        self._reposition_points()
        half_x, half_y = float(self.points[1, 0] - self.points[0, 0]) / 2, float(self.points[1, 1] - self.points[0, 1]) / 2
        self.vel[0] = linear_vel_avg + angular_vel * np.array([half_y, -half_x])  # rigid: centre velocity -/+ spin x half-stick
        self.vel[1] = linear_vel_avg - angular_vel * np.array([half_y, -half_x])
        self.angular_velocity = angular_vel
        self.linear_velocity_magnitude = np.linalg.norm(linear_vel_avg)

    def _reposition_points(self):
        half_size_x = self.size * math.cos(self.angle) / 2
        half_size_y = self.size * math.sin(self.angle) / 2
        center_x, center_y = float(self.center[0]), float(self.center[1])
        self.points[0, 0] = center_x - half_size_x
        self.points[0, 1] = center_y - half_size_y
        self.points[1, 0] = center_x + half_size_x
        self.points[1, 1] = center_y + half_size_y

    def update(self, gravity_constant, delta_time_step):
        self.previous_linear_velocity_magnitude = self.linear_velocity_magnitude
        current_angle_for_angular_vel_calc = self.angle
        if self.drag:
            self.vel *= (1 - self.drag)
        self.vel[:, 1] += gravity_constant * delta_time_step
        self.points += self.vel * delta_time_step
        self.center = (self.points[0] + self.points[1]) / 2
        self.angle = normalize_angle_signed(math.atan2(self.points[1, 1] - self.points[0, 1], self.points[1, 0] - self.points[0, 0]))
        angle_diff = normalize_angle_signed(self.angle - current_angle_for_angular_vel_calc)
        self.angular_velocity = angle_diff / delta_time_step if delta_time_step > 0 else 0.0
        self.linear_velocity_magnitude = math.hypot(float(self.vel[0, 0] + self.vel[1, 0]) / 2, float(self.vel[0, 1] + self.vel[1, 1]) / 2)
        self._reposition_points()


class Drone:
    def __init__(self, initial_pos, size, envDiag):
        self.starting_pos = initial_pos
        self.size = size
        self.envDiag = envDiag
        self.droneSpeedScalar = DRONE_MAX_SPEED_SCALAR
        self.droneRotationScalar = DRONE_MAX_ROT_SCALAR
        self.shell = stick(self.size)
        self.maxThrusterForce = MAX_THRUSTER_FORCE
        self.thruster_forces = [0.0, 0.0]
        self.previous_thruster_forces = [0.0, 0.0]
        self.thruster_force_delta_rate = THRUSTER_FORCE_DELTA_RATE
        self.thruster_angles = [THRUSTER_NEUTRAL_ANGLE, THRUSTER_NEUTRAL_ANGLE]  # relative to the drone body
        self.thruster_angle_delta_rate = THRUSTER_ANGLE_DELTA_RATE
        self.currentObservationState = None

    def reset(self, initial_pos=None):
        actual_initial_pos = initial_pos if initial_pos is not None else self.starting_pos
        random_angle = random.uniform(-math.pi / 4, math.pi / 4)
        random_linear_vel_x = random.uniform(-INITIAL_MAX_LINEAR_VEL, INITIAL_MAX_LINEAR_VEL)
        random_linear_vel_y = random.uniform(-INITIAL_MAX_LINEAR_VEL, INITIAL_MAX_LINEAR_VEL)
        random_angular_vel = random.uniform(-INITIAL_MAX_ANGULAR_VEL, INITIAL_MAX_ANGULAR_VEL)
        self.shell.set_physical_state(actual_initial_pos, random_angle, np.array([random_linear_vel_x, random_linear_vel_y], dtype=np.float32), random_angular_vel)
        self.thruster_forces = [0.0, 0.0]
        self.previous_thruster_forces = [0.0, 0.0]
        self.thruster_angles = [THRUSTER_NEUTRAL_ANGLE, THRUSTER_NEUTRAL_ANGLE]
        self.currentObservationState = np.zeros(OBS_DIM)

    def get_observation(self, target_pos, extra=()):
        """The drone's 17 observations, followed by any extra values the caller wants appended."""
        center = self.shell.center
        drone_x, drone_y = float(center[0]), float(center[1])
        vel = self.shell.vel
        vel_x = (float(vel[0, 0]) + float(vel[1, 0])) / 2
        vel_y = (float(vel[0, 1]) + float(vel[1, 1])) / 2
        to_target_x, to_target_y = float(target_pos[0]) - drone_x, float(target_pos[1]) - drone_y
        target_dist = math.hypot(to_target_x, to_target_y)
        velocity_towards_target = (vel_x * to_target_x + vel_y * to_target_y) / (target_dist + 1e-8)

        normalized_h_dist = min(drone_x, ENV_WIDTH - drone_x) / (ENV_WIDTH / 2.0)
        normalized_v_dist = min(drone_y, ENV_HEIGHT - drone_y) / (ENV_HEIGHT / 2.0)

        # --- Time-to-Impact Calculation (3 s is the safe default when barely moving) ---
        h_tti = 3.0
        if vel_x > 1.0:
            h_tti = (ENV_WIDTH - drone_x) / vel_x
        elif vel_x < -1.0:
            h_tti = drone_x / -vel_x
        v_tti = 3.0
        if vel_y > 1.0:
            v_tti = (ENV_HEIGHT - drone_y) / vel_y
        elif vel_y < -1.0:
            v_tti = drone_y / -vel_y

        # Inverse TTI through tanh gives a bounded value that approaches 1 as impact nears and 0 for large TTI
        tanh = math.tanh
        angle = self.shell.angle
        observation = np.array(
            [tanh(vel_x / self.droneSpeedScalar), tanh(vel_y / self.droneSpeedScalar), tanh(self.shell.angular_velocity / self.droneRotationScalar),
             tanh(self.shell.linear_velocity_magnitude / (self.droneSpeedScalar * math.sqrt(2))), to_target_x / self.envDiag, to_target_y / self.envDiag,
             self.thruster_forces[0], self.thruster_forces[1],
             (self.thruster_angles[0] - THRUSTER_NEUTRAL_ANGLE) / THRUSTER_ANGLE_RANGE, (self.thruster_angles[1] - THRUSTER_NEUTRAL_ANGLE) / THRUSTER_ANGLE_RANGE,
             math.sin(angle), math.cos(angle), tanh(velocity_towards_target / self.droneSpeedScalar), normalized_h_dist, normalized_v_dist,
             tanh(1.0 / (h_tti + 1e-6)), tanh(1.0 / (v_tti + 1e-6)), *extra], dtype=np.float32)

        self.currentObservationState = observation
        return observation

    def apply_action(self, action, delta_time_step):
        self.previous_thruster_forces = list(self.thruster_forces)
        self.thruster_forces[0] = clamp(0, 1, self.thruster_forces[0] + action[0] * self.thruster_force_delta_rate)
        self.thruster_forces[1] = clamp(0, 1, self.thruster_forces[1] + action[1] * self.thruster_force_delta_rate)
        for i in range(2):
            self.thruster_angles[i] = clamp(*THRUSTER_ANGLE_LIMITS[i], self.thruster_angles[i] + action[2 + i] * self.thruster_angle_delta_rate)

    def _make_rigid(self):
        """Keeps only the point velocities a rigid stick could have (centre velocity plus spin), dropping the stretch/squeeze velocity that sideways thrust adds."""
        p, v = self.shell.points, self.shell.vel
        rx, ry = float(p[1, 0] - p[0, 0]), float(p[1, 1] - p[0, 1])
        rel_x, rel_y = float(v[1, 0] - v[0, 0]), float(v[1, 1] - v[0, 1])
        omega = (rx * rel_y - ry * rel_x) / (rx * rx + ry * ry)
        center_x, center_y = float(v[0, 0] + v[1, 0]) / 2, float(v[0, 1] + v[1, 1]) / 2
        half_x, half_y = rx / 2, ry / 2
        v[0, 0], v[0, 1] = center_x + omega * half_y, center_y - omega * half_x
        v[1, 0], v[1, 1] = center_x - omega * half_y, center_y + omega * half_x

    def physics_move(self, gravity, delta_time_step):
        thruster_1_absolute_angle = normalize_angle_signed(self.thruster_angles[0] + self.shell.angle)
        thruster_2_absolute_angle = normalize_angle_signed(self.thruster_angles[1] + self.shell.angle)
        vel = self.shell.vel
        scale_1 = self.thruster_forces[0] * delta_time_step * self.maxThrusterForce
        scale_2 = self.thruster_forces[1] * delta_time_step * self.maxThrusterForce
        vel[0, 0] -= scale_1 * math.cos(thruster_1_absolute_angle)
        vel[0, 1] -= scale_1 * math.sin(thruster_1_absolute_angle)
        vel[1, 0] -= scale_2 * math.cos(thruster_2_absolute_angle)
        vel[1, 1] -= scale_2 * math.sin(thruster_2_absolute_angle)
        self._make_rigid()
        self.shell.update(gravity, delta_time_step)

    def draw(self, pygame_lib, screen, colMain, colSecondary, thruster_draw_size=5):
        p0_int = [int(self.shell.points[0][0]), int(self.shell.points[0][1])]
        p1_int = [int(self.shell.points[1][0]), int(self.shell.points[1][1])]

        pygame_lib.draw.line(screen, colMain, p0_int, p1_int, 2)

        pygame_lib.draw.circle(screen, colSecondary, p0_int, 3)
        pygame_lib.draw.circle(screen, colSecondary, p1_int, 3)

        thruster1_abs_draw_angle = normalize_angle_signed(self.thruster_angles[0] + self.shell.angle)
        thruster1_end = [
            int(self.shell.points[0][0] + thruster_draw_size * math.cos(thruster1_abs_draw_angle)),
            int(self.shell.points[0][1] + thruster_draw_size * math.sin(thruster1_abs_draw_angle))
        ]
        pygame_lib.draw.line(screen, colSecondary, p0_int, thruster1_end, 2)

        thruster2_abs_draw_angle = normalize_angle_signed(self.thruster_angles[1] + self.shell.angle)
        thruster2_end = [
            int(self.shell.points[1][0] + thruster_draw_size * math.cos(thruster2_abs_draw_angle)),
            int(self.shell.points[1][1] + thruster_draw_size * math.sin(thruster2_abs_draw_angle))
        ]
        pygame_lib.draw.line(screen, colSecondary, p1_int, thruster2_end, 2)