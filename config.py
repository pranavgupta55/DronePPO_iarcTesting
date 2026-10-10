import math

# --- Physics Constants ---
METERS_PER_PIXEL = 10.0
PHYSICS_FPS = 50
TIME_STEP_DELTA = 1.0 / PHYSICS_FPS
GRAVITY = 50.0 * METERS_PER_PIXEL

# --- Environment Dimensions ---
ENV_WIDTH = 1500
ENV_HEIGHT = 1000
ENV_DIAG = (ENV_WIDTH ** 2 + ENV_HEIGHT ** 2) ** 0.5
DRONE_SIZE = 10

# --- Target Spawning ---
TARGET_MARGIN = 100  # targets spawn at least this far from every edge
# Far-target curriculum: a growing fraction of new targets must be at least FAR_TARGET_MIN_DIST from the drone. The fraction goes from 0 to
# FAR_TARGET_FRACTION_END over the first TARGET_CURRICULUM_END of the training session; the rest of the targets stay uniform over the arena.
FAR_TARGET_MIN_DIST = 600.0
FAR_TARGET_FRACTION_END = 0.7
TARGET_CURRICULUM_END = 0.5

# --- Observation / Action Space ---
abbreviatedStateNames = [
    "velX", "velY", "angVel", "velMag", "relX", "relY",
    "T1Force", "T2Force", "T1Angle", "T2Angle", "sinAng", "cosAng", "velToTarget",
    "hDistWall", "vDistWall", "h_TTI", "v_TTI", "tiltWeight"
]
OBS_DIM = 18  # the drone's 17 observations plus the tilt penalty weight (added by DroneEnv.observe)
ACT_DIM = 4  # [thruster 1 force change, thruster 2 force change, thruster 1 angle change, thruster 2 angle change]

# --- Drone Specifics ---
MAX_THRUSTER_FORCE = 6000.0
THRUSTER_FORCE_DELTA_RATE = 0.005  # force change per step per unit action (actions are not clipped; ~1 is a full-speed change)
DRONE_MAX_SPEED_SCALAR = 300.0
DRONE_MAX_ROT_SCALAR = 10.0
THRUSTER_NEUTRAL_ANGLE = math.pi / 2  # thruster direction relative to the drone body when it points straight "up"
THRUSTER_ANGLE_RANGE = math.pi / 2  # how far each thruster can swivel outward from neutral (90 degrees)
THRUSTER_INWARD_ALLOWANCE = math.radians(30)  # and how far it may swivel inward, which keeps some steering authority
THRUSTER_ANGLE_LIMITS = [(THRUSTER_NEUTRAL_ANGLE - THRUSTER_INWARD_ALLOWANCE, THRUSTER_NEUTRAL_ANGLE + THRUSTER_ANGLE_RANGE),  # left thruster: a little inward to bottom-left
                         (THRUSTER_NEUTRAL_ANGLE - THRUSTER_ANGLE_RANGE, THRUSTER_NEUTRAL_ANGLE + THRUSTER_INWARD_ALLOWANCE)]  # right thruster: bottom-right to a little inward
THRUSTER_ANGLE_DELTA_RATE = math.pi / 60  # radians per step per unit action (an action of 1.0 turns 3 degrees per step)

# --- Initial Random State Configuration ---
INITIAL_MAX_LINEAR_VEL = 20.0
INITIAL_MAX_ANGULAR_VEL = math.radians(60)

# --- Rewards (per step unless noted) ---
DISTANCE_REWARD = 0.3  # times (1 - distance_to_target / ENV_DIAG), so it is linear in distance
VELOCITY_REWARD = 1.0  # times the drone's speed toward the target divided by VELOCITY_REFERENCE (clipped to +/-1), so it pays for actually approaching, not just drifting the right way
VELOCITY_REFERENCE = 100.0  # px/s of approach speed that earns the full velocity reward
ZONE_REWARD = 0.5  # while within REACH_DIST of the target
OUTWARD_REWARD = 1.5  # max reward for pointing the thruster nozzles outwards (left one bottom-left, right one bottom-right); each nozzle scores 1 - |outward angle - OUTWARD_TARGET| / OUTWARD_TARGET, so 0 at neutral and negative when pointing inward
OUTWARD_TARGET = math.pi / 6  # nozzle angle from straight down that earns the max; peaked rather than "as far as possible" so it doesn't reward pinning the angle at its limit
# The caps on the level reward and the tilt penalty move over the training session: strict at the start, so the drone first learns to fly
# level, then relaxed so it can tilt to move fast. They reach their END values REWARD_RAMP_END of the way through the session.
REWARD_RAMP_END = 0.5
# Tilt penalty: per step, -weight * |body angle| / pi. The weight starts at TILT_PENALTY_MIN each episode, ramps up slowly while the
# drone stays non-level (|angle| > LEVEL_ANGLE), up to the cap, and falls quickly back toward the minimum once it is level again.
LEVEL_ANGLE = math.radians(10)
TILT_PENALTY_MIN = 0.005
TILT_PENALTY_MAX_START = 4.0
TILT_PENALTY_MAX_END = 2.0
TILT_PENALTY_RAMP_STEPS = 1800  # non-level steps for the weight to climb from the minimum to the cap
# Tilting is only needed to move fast, so the tilt penalty is multiplied by (1 + SLOW_TILT_EXTRA * (1 - speed / SLOW_TILT_SPEED)) while slower than SLOW_TILT_SPEED.
SLOW_TILT_EXTRA = 2.0
SLOW_TILT_SPEED = 100.0  # px/s
TILT_PENALTY_DECAY = 0.8  # weight is multiplied by this each level step (about 7% of its value after 12 steps)
# Level reward: per step while level, +weight * (1 - |angle| / LEVEL_ANGLE), so exactly level pays the most. The weight ramps up
# (a long level streak pays more) and is multiplied by LEVEL_REWARD_DECAY each non-level step.
LEVEL_REWARD_MAX_START = 1.0
LEVEL_REWARD_MAX_END = 0.25
LEVEL_REWARD_RAMP_STEPS = 300  # level steps for the weight to climb from 0 to the cap
LEVEL_REWARD_DECAY = 0.5
# Angular velocity penalty: per step, -weight * (|angular velocity| - deadband) while the spin (rad/s) is above the deadband.
ANGULAR_VELOCITY_PENALTY = 0.1
ANGULAR_VELOCITY_DEADBAND = 0.3
# Smoothness penalties, kept small and phased in late in the session (from 0 at SMOOTHNESS_RAMP_START of the way through to full at
# SMOOTHNESS_RAMP_END), so the drone first learns to fly well and only then to fly smoothly.
SMOOTHNESS_RAMP_START = 0.3
SMOOTHNESS_RAMP_END = 0.7
ACTION_CHANGE_PENALTY = 0.2  # times the mean |action - previous action|
ANGULAR_ACCEL_PENALTY = 0.2  # times |change in angular velocity| per step (rad/s)
LATERAL_VELOCITY_PENALTY = 0.3  # times min((lateral speed - LATERAL_VELOCITY_DEADBAND) / LATERAL_VELOCITY_REFERENCE, 2): speed perpendicular to the line to the target
LATERAL_VELOCITY_DEADBAND = 20.0  # px/s
LATERAL_VELOCITY_REFERENCE = 100.0  # px/s
REACH_REWARD = 1000.0  # once per target reached (about 1000 steps of the standing rewards, so reaching has to beat hovering)
CRASH_PENALTY = 300.0  # once, when the drone leaves the world

# --- Target Reach Conditions ---
REACH_DIST = 20.0  # center-to-target distance below which the target counts as reached
# And the drone must be nearly still and not spinning, so it arrives calmly and hovers. These limits start loose and tighten to their END values
# over the curriculum (like the level and tilt caps), so early learning isn't blocked.
REACH_SPEED_START = 50.0  # px/s
REACH_SPEED_END = 20.0
REACH_ANGULAR_SPEED_START = 2.0  # rad/s
REACH_ANGULAR_SPEED_END = 0.5
# It must also meet all of those for this many steps in a row (50 steps = 1 second) before the target counts as reached; this also ramps up.
REACH_HOLD_STEPS_START = 1
REACH_HOLD_STEPS_END = 25
