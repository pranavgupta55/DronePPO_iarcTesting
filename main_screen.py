import json
import sys
import pygame
import math
import time
import random
from pathlib import Path
import numpy as np
import torch
from text import drawText, wrapText
from fontDict import fonts
from network import NeuralNetwork, draw_network, tanh, tanh_prime
from ppo import Actor, DroneEnv, RunningNorm, REACH_DIST
from config import OBS_DIM, ACT_DIM, ENV_WIDTH, ENV_HEIGHT, TIME_STEP_DELTA, THRUSTER_ANGLE_LIMITS, TARGET_MARGIN

pygame.init()

# ---------------- Setting up the screen, assigning some global variables, and loading text fonts
screen = pygame.display.set_mode((1512, 982), pygame.FULLSCREEN)
clock = pygame.time.Clock()
fps = 60
screen_width = screen.get_width()
screen_height = screen.get_height()
screen_center = [screen_width / 2, screen_height / 2]
screen2 = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT.set_alpha(100)
screenUI = pygame.Surface((screen_width, screen_height)).convert_alpha()
timer = 0
shake = [0, 0]
shake_strength = 3
montserratRegularAdaptive = fonts["regular25"]
montserratExtralightAdaptive = fonts["extralight25"]
montserratBoldAdaptive = fonts["bold25"]
montserratThinAdaptive = fonts["thin25"]


class Endesga:
    maroon_red = [87, 28, 39]
    lighter_maroon_red = [127, 36, 51]
    dark_green = [9, 26, 23]
    light_brown = [191, 111, 74]
    black = [19, 19, 19]
    grey_blue = [66, 76, 110]
    cream = [237, 171, 80]
    white = [255, 255, 255]
    greyL = [200, 200, 200]
    grey = [150, 150, 150]
    greyD = [100, 100, 100]
    greyVD = [50, 50, 50]
    very_light_blue = [199, 207, 221]
    my_blue = [32, 36, 46]
    debug_red = [255, 96, 141]
    sebastian_lague_purple = [70, 74, 124]
    sebastian_lague_light_purple = [137, 133, 181]
    network_green = [64, 128, 67]
    network_red = [127, 45, 41]


class FireCols:
    cols = [[255, 242, 157], [255, 212, 124], [255, 169, 70], [254, 128, 3], [230, 106, 37], [179, 46, 19], [126, 19, 6], [70, 2, 14]]
    colsReversed = [[70, 2, 14], [126, 19, 6], [179, 46, 19], [230, 106, 37], [254, 128, 3], [255, 169, 70], [255, 212, 124], [255, 242, 157]]


# Defining some more variables to use in the game loop
oscillating_random_thing = 0
ShakeCounter = 0
toggle = True
click = False


RUN_INDEX = 38  # which runs/run<N>/actor.pt to fly
EXPERIMENT_RUN = "b015/t4_warm60"  # "bc/bc_feat" is the pure imitation network, "b011/ft_gentle_lr1e-05" its PPO fine-tune; a run under experiments/runs/<batch>/<name> to fly instead (the world is the experiments one, 2400x1600); "bc/bc_feat" is the network imitating the bang-bang controller; None flies RUN_INDEX
BANG_BANG = False  # fly the hand-written bang-bang controller (experiments/bangbang_controller.py, nozzles neutral) instead of a trained network; uses the 1200x800 experiment world
EPISODE_SECONDS = 500


def load_run(run_index):
    """Returns a fresh DroneEnv, the actor saved by training run <run_index>, and its observation normalizer (None for runs saved before normalization existed)."""
    run_dir = Path(__file__).parent / "runs" / f"run{run_index}"
    state = torch.load(run_dir / "actor.pt")
    hidden_sizes = [w.shape[0] for k, w in state.items() if k.startswith("net.") and k.endswith("weight")][:-1]  # layer sizes, minus the output layer
    actor = Actor(hidden_sizes)
    actor.load_state_dict(state)
    obs_norm = None
    if (run_dir / "norm.npz").exists():
        norms = np.load(run_dir / "norm.npz")
        obs_norm = RunningNorm((OBS_DIM,))
        obs_norm.mean, obs_norm.var = norms["obs_mean"], norms["obs_var"]
    return DroneEnv(max_steps=int(EPISODE_SECONDS / TIME_STEP_DELTA), mirror_episodes=False), actor, obs_norm


def load_experiment_run(name):
    """Returns a fresh experiments DroneEnv (strict reach rule, no mirroring), the run's (weight, bias) layers, an observation normalizer function and the world size."""
    experiments_dir = Path(__file__).parent / "experiments"
    sys.path.insert(0, str(experiments_dir))
    from core.env import DroneEnv as ExperimentEnv, make_spec, load_scheme, NORM_CLIP
    from core.config import ENV_WIDTH as exp_width, ENV_HEIGHT as exp_height
    from evaluate import load_policy
    run_dir = experiments_dir / "runs" / name
    spec = make_spec(json.loads((run_dir / "spec.json").read_text()))
    layers, obs_mean, obs_std = load_policy(run_dir)
    module, scheme = load_scheme(spec["scheme"], spec["params"])
    exp_env = ExperimentEnv(spec, module, scheme, max_steps=int(EPISODE_SECONDS / TIME_STEP_DELTA))
    exp_env.set_ramp(1.0)
    exp_env.set_eval_mode()
    if not spec["obs_norm"]:
        obs_mean, obs_std = np.zeros_like(obs_mean), np.ones_like(obs_std)
    return exp_env, layers, (lambda o: np.clip((o - obs_mean) / obs_std, -NORM_CLIP, NORM_CLIP)), exp_width, exp_height


def layers_to_network(layers):
    """Copies (weight, bias) layers into a network.py NeuralNetwork so draw_network can draw it."""
    network = NeuralNetwork([layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers], tanh, tanh_prime)
    for dense, (weight, bias) in zip((layer for layer in network.layers if hasattr(layer, "weights")), layers):
        dense.weights = np.asarray(weight, dtype=float)
        dense.bias = np.asarray(bias, dtype=float).reshape(-1, 1)
    return network


def draw_drone(surface, drone, scale):
    """Same picture as Drone.draw, with the positions scaled into the zoomed-out view (line widths and marker sizes stay in screen pixels)."""
    points = [(float(point[0]) * scale, float(point[1]) * scale) for point in drone.shell.points]
    pygame.draw.line(surface, Endesga.white, points[0], points[1], 2)
    for point, angle in zip(points, drone.thruster_angles):
        absolute = angle + drone.shell.angle
        pygame.draw.circle(surface, Endesga.debug_red, point, 3)
        pygame.draw.line(surface, Endesga.debug_red, point, (point[0] + 10 * math.cos(absolute), point[1] + 10 * math.sin(absolute)), 2)


def run_label():
    """(short name, blurb) of the model being flown: the first line of its blurb.txt, then the rest. Runs without a blurb show their path."""
    if BANG_BANG:
        return "Bang-bang controller", "Hand-written controller, not a trained network. Flips, thrusts hard, flips back to brake."
    run_dir = Path(__file__).parent / "experiments" / "runs" / EXPERIMENT_RUN if EXPERIMENT_RUN else Path(__file__).parent / "runs" / f"run{RUN_INDEX}"
    label = EXPERIMENT_RUN or f"run{RUN_INDEX}"
    if (run_dir / "blurb.txt").exists():
        lines = (run_dir / "blurb.txt").read_text().strip().splitlines()
        return f"{label}: {lines[0]}", " ".join(lines[1:])
    return label, "(no blurb)"


GAUGE_OUTER, GAUGE_INNER, GAUGE_GAP, GAUGE_MARGIN = 56, 18, 20, 12  # thruster gauge: outer / inner radius, gap between the two sectors, distance from the screen's corner (px)


def draw_thruster_gauge(surface, drone):
    """Top right: one downward-facing annulus sector per thruster covering its swivel range (left thruster on the left), with a red line along the thruster's
    current direction relative to the drone body; the line runs from the inner arc out to a radius proportional to the thruster's force (the outer arc is full force)."""
    right_center_x = screen_width - GAUGE_MARGIN - GAUGE_OUTER  # the right sector reaches one outer radius to the right of its center
    center_y = GAUGE_MARGIN
    centers = [(right_center_x - GAUGE_OUTER - GAUGE_GAP, center_y), (right_center_x, center_y)]
    for (center_x, center_y_), (low, high), angle, force in zip(centers, THRUSTER_ANGLE_LIMITS, drone.thruster_angles, drone.thruster_forces):
        steps = 24
        angles = [low + (high - low) * i / steps for i in range(steps + 1)]
        outer = [(center_x + GAUGE_OUTER * math.cos(a), center_y_ + GAUGE_OUTER * math.sin(a)) for a in angles]
        inner = [(center_x + GAUGE_INNER * math.cos(a), center_y_ + GAUGE_INNER * math.sin(a)) for a in angles]
        pygame.draw.lines(surface, Endesga.white, False, outer, 1)
        pygame.draw.lines(surface, Endesga.white, False, inner, 1)
        pygame.draw.line(surface, Endesga.white, outer[0], inner[0], 1)
        pygame.draw.line(surface, Endesga.white, outer[-1], inner[-1], 1)
        length = GAUGE_INNER + force * (GAUGE_OUTER - GAUGE_INNER)
        pygame.draw.line(surface, Endesga.debug_red, (center_x + GAUGE_INNER * math.cos(angle), center_y_ + GAUGE_INNER * math.sin(angle)),
                         (center_x + length * math.cos(angle), center_y_ + length * math.sin(angle)), 4)


if BANG_BANG:
    env, layers, normalize, WORLD_W, WORLD_H = load_experiment_run("b003/time_to_wall_barrier_x40_out30")  # only for its environment; the network is not used
    layers = None
    from bangbang_controller import control as bang_bang_control, DEFAULTS as BANG_BANG_PARAMS

    def policy(o):
        return bang_bang_control(env, BANG_BANG_PARAMS)
elif EXPERIMENT_RUN:
    env, layers, normalize, WORLD_W, WORLD_H = load_experiment_run(EXPERIMENT_RUN)

    def policy(o):
        x = normalize(o)
        for weight, bias in layers[:-1]:
            x = np.tanh(weight @ x + bias)
        return layers[-1][0] @ x + layers[-1][1]  # the mean action, no sampling noise
else:
    env, actor, obs_norm = load_run(RUN_INDEX)
    WORLD_W, WORLD_H = ENV_WIDTH, ENV_HEIGHT
    layers = [(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in actor.net if isinstance(m, torch.nn.Linear)]

    def policy(o):
        with torch.no_grad():
            return actor.net(torch.as_tensor(o if obs_norm is None else obs_norm.normalize(o), dtype=torch.float32)).numpy()  # the mean action, no sampling noise
obs = env.reset()
view_scale = min(screen_width / WORLD_W, screen_height / WORLD_H, 1.0)  # shrink the world to fit the screen, never enlarge it
world_size = (int(WORLD_W * view_scale), int(WORLD_H * view_scale))
world = pygame.Surface(world_size)  # drawn at screen size directly (a zoomed-out world is never rendered at full size, so the frame rate doesn't depend on the world size)
world_offset = ((screen_width - world_size[0]) // 2, (screen_height - world_size[1]) // 2)

# The weights are fixed, so draw the network once onto its own surface (thousands of lines) and blit it every frame
network_surface, network_corner = None, (0, 0)
if layers is not None:
    network = layers_to_network(layers)
    dense_layers = [layer for layer in network.layers if hasattr(layer, "weights")]
    tallest_layer = max(max(layer.weights.shape) for layer in dense_layers)
    network_surface = pygame.Surface((20 + 60 * len(dense_layers), 200), pygame.SRCALPHA)
    draw_network(network, network_surface, (10, 100), Endesga.network_green, Endesga.network_red, spacing_x=60, spacing_y=min(3, 180 / tallest_layer))
    network_bounds = network_surface.get_bounding_rect()  # where the drawing actually is, so its top left corner can sit at the screen's corner margin
    network_corner = (GAUGE_MARGIN - network_bounds.x, GAUGE_MARGIN - network_bounds.y)
physics_accumulator = 0.0
run_name, run_blurb = run_label()

# ---------------- Main Game Loop
last_time = time.time()
running = True
while running:

    # ---------------- Reset Variables and Clear screens
    mx, my = pygame.mouse.get_pos()
    screen.fill(Endesga.my_blue)
    screen2.fill(Endesga.my_blue)
    screenT.fill((0, 0, 0, 0))
    screenUI.fill((0, 0, 0, 0))
    dt = time.time() - last_time
    dt *= fps
    last_time = time.time()
    timer -= 1 * dt
    shake = [0, 0]
    oscillating_random_thing += math.pi / fps * dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        if event.type == pygame.MOUSEBUTTONUP:
            click = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                toggle = not toggle
        if event.type == pygame.KEYUP:
            pass

    # ---------------- Drone (fixed physics timestep; dt is in frame units, so dt / fps is real seconds)
    physics_accumulator = min(physics_accumulator + dt / fps, 0.25)
    mouse_target = np.array([(mx - world_offset[0]) / view_scale, (my - world_offset[1]) / view_scale])  # the mouse is the target, in world coordinates
    mouse_target = np.clip(mouse_target, TARGET_MARGIN, [WORLD_W - TARGET_MARGIN, WORLD_H - TARGET_MARGIN])  # a target in or beyond a wall is outside what the drone was built for (it flew into the wall and crashed)
    env.target = mouse_target
    obs = env.observe()
    while physics_accumulator >= TIME_STEP_DELTA:
        obs, _, terminated, truncated = env.step(policy(obs))
        if terminated or truncated:
            env.reset()
        env.target = mouse_target  # a reach or a reset spawns a random target; the mouse overrides it
        obs = env.observe()
        physics_accumulator -= TIME_STEP_DELTA

    world.fill(Endesga.my_blue)
    if toggle:
        pygame.draw.rect(world, Endesga.black, world.get_rect(), 1)  # world bounds
        pygame.draw.circle(world, Endesga.cream, env.target * view_scale, max(REACH_DIST * view_scale, 3), 1)
    draw_drone(world, env.drone, view_scale)
    screen2.blit(world, world_offset)

    # ---------------- Updating Screen
    if toggle:
        items = {round(clock.get_fps()): None,
                 }
        for i, label in enumerate(items.keys()):
            string = str(label)
            if items[label] is not None:
                string = f"{items[label]}: " + string
            drawText(screenUI, Endesga.debug_red, montserratRegularAdaptive, 5, screen_height - (30 + 25 * i), string, Endesga.black, 0, antiAliasing=False)

        # ---------------- Network
        if network_surface is not None:
            screenUI.blit(network_surface, network_corner)
        draw_thruster_gauge(screenUI, env.drone)
        label_font, blurb_font = fonts["bold20"], fonts["regular16"]
        blurb_lines, _ = wrapText(blurb_font, run_blurb, 460)
        y = screen_height - 12 - len(blurb_lines) * blurb_font.get_height() * 1.1 - label_font.get_height() * 1.2
        drawText(screenUI, Endesga.white, label_font, screen_width - 12, y, run_name, Endesga.black, 1, justify="right")
        drawText(screenUI, Endesga.greyL, blurb_font, screen_width - 12, y + label_font.get_height() * 1.2, run_blurb, Endesga.black, 1, wrap=True, maxLen=460, justify="right")
    pygame.mouse.set_visible(False)
    pygame.draw.circle(screenUI, Endesga.black, (mx + 1, my + 1), 2, 1)
    pygame.draw.circle(screenUI, Endesga.white, (mx, my), 2, 1)
    screen.blit(screen2, (shake[0], shake[1]))
    screen.blit(screenT, (shake[0], shake[1]))
    screen.blit(screenUI, (0, 0))
    pygame.display.update()
    clock.tick(fps)
