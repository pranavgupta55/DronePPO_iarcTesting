import math

# Fixed environment for every experiment (physics copied from the project's config.py; rewards live in the schemes, not here).

# --- Physics Constants ---
METERS_PER_PIXEL = 10.0
PHYSICS_FPS = 50
TIME_STEP_DELTA = 1.0 / PHYSICS_FPS
GRAVITY = 50.0 * METERS_PER_PIXEL

# --- Environment Dimensions ---
ENV_WIDTH = 2400
ENV_HEIGHT = 1600
ENV_DIAG = (ENV_WIDTH ** 2 + ENV_HEIGHT ** 2) ** 0.5
DRONE_SIZE = 10

# --- Observation / Action Space ---
OBS_BASE_DIM = 17  # the drone's own observations (Drone.get_observation); a scheme may append 0..MAX_EXTRA_OBS extra features
MAX_EXTRA_OBS = 7
ACT_DIM = 4  # [thruster 1 force change, thruster 2 force change, thruster 1 angle change, thruster 2 angle change]

# --- Drone Specifics ---
MAX_THRUSTER_FORCE = 6000.0
THRUSTER_FORCE_DELTA_RATE = 0.005
DRONE_MAX_SPEED_SCALAR = 300.0
DRONE_MAX_ROT_SCALAR = 10.0
THRUSTER_NEUTRAL_ANGLE = math.pi / 2
THRUSTER_ANGLE_RANGE = math.pi / 2
THRUSTER_INWARD_ALLOWANCE = math.radians(30)
THRUSTER_ANGLE_LIMITS = [(THRUSTER_NEUTRAL_ANGLE - THRUSTER_INWARD_ALLOWANCE, THRUSTER_NEUTRAL_ANGLE + THRUSTER_ANGLE_RANGE),  # left thruster
                         (THRUSTER_NEUTRAL_ANGLE - THRUSTER_ANGLE_RANGE, THRUSTER_NEUTRAL_ANGLE + THRUSTER_INWARD_ALLOWANCE)]  # right thruster
THRUSTER_ANGLE_DELTA_RATE = math.pi / 60

# --- Initial Random State Configuration ---
INITIAL_MAX_LINEAR_VEL = 20.0
INITIAL_MAX_ANGULAR_VEL = math.radians(60)

# --- Time-optimal estimate (used for ctx.time_to_go) ---
TIME_OPTIMAL_ACCEL = 6000.0  # px/s^2

# --- Strict reach rule (used by evaluation) ---
REACH_DIST = 20.0
STRICT_REACH_SPEED = 20.0  # px/s
STRICT_REACH_ANG = 0.5  # rad/s
STRICT_REACH_HOLD = 25  # consecutive steps
