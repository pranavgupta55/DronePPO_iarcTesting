import pygame
import math
import time
import random
from pathlib import Path
import numpy as np
import torch
from text import drawText
from fontDict import fonts
from network import NeuralNetwork, draw_network, tanh, tanh_prime
from ppo import Actor, DroneEnv, RunningNorm, REACH_DIST
from config import OBS_DIM, ACT_DIM, ENV_WIDTH, ENV_HEIGHT, TIME_STEP_DELTA

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


RUN_INDEX = 38
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


def actor_to_network(actor):
    """Copies the actor's linear layers into a network.py NeuralNetwork so draw_network can draw it."""
    linears = [m for m in actor.net if isinstance(m, torch.nn.Linear)]
    network = NeuralNetwork([OBS_DIM] + [m.out_features for m in linears], tanh, tanh_prime)
    for dense, linear in zip((layer for layer in network.layers if hasattr(layer, "weights")), linears):
        dense.weights = linear.weight.detach().numpy().astype(float)
        dense.bias = linear.bias.detach().numpy().astype(float).reshape(-1, 1)
    return network


env, actor, obs_norm = load_run(RUN_INDEX)
obs = env.reset()
world = pygame.Surface((ENV_WIDTH, ENV_HEIGHT))
view_scale = min(screen_width / ENV_WIDTH, screen_height / ENV_HEIGHT, 1.0)  # shrink the world to fit the screen, never enlarge it
world_size = (int(ENV_WIDTH * view_scale), int(ENV_HEIGHT * view_scale))
world_offset = ((screen_width - world_size[0]) // 2, (screen_height - world_size[1]) // 2)

# The weights are fixed, so draw the network once onto its own surface (thousands of lines) and blit it every frame
network = actor_to_network(actor)
dense_layers = [layer for layer in network.layers if hasattr(layer, "weights")]
tallest_layer = max(max(layer.weights.shape) for layer in dense_layers)
network_surface = pygame.Surface((20 + 60 * len(dense_layers), 200), pygame.SRCALPHA)
draw_network(network, network_surface, (10, 100), Endesga.network_green, Endesga.network_red, spacing_x=60, spacing_y=min(3, 180 / tallest_layer))
physics_accumulator = 0.0

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
    env.target = mouse_target
    obs = env.observe()
    while physics_accumulator >= TIME_STEP_DELTA:
        with torch.no_grad():
            action = actor.net(torch.as_tensor(obs if obs_norm is None else obs_norm.normalize(obs), dtype=torch.float32)).numpy()  # the mean action, no sampling noise
        obs, _, terminated, truncated = env.step(action)
        if terminated or truncated:
            env.reset()
        env.target = mouse_target  # a reach or a reset spawns a random target; the mouse overrides it
        obs = env.observe()
        physics_accumulator -= TIME_STEP_DELTA

    world.fill(Endesga.my_blue)
    if toggle:
        pygame.draw.rect(world, Endesga.black, world.get_rect(), 1)  # world bounds
        pygame.draw.circle(world, Endesga.cream, env.target, REACH_DIST, 1)
    env.drone.draw(pygame, world, Endesga.white, Endesga.debug_red, 10)
    screen2.blit(world if view_scale == 1.0 else pygame.transform.smoothscale(world, world_size), world_offset)

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
        screenUI.blit(network_surface, (0, 0))
    pygame.mouse.set_visible(False)
    pygame.draw.circle(screenUI, Endesga.black, (mx + 1, my + 1), 2, 1)
    pygame.draw.circle(screenUI, Endesga.white, (mx, my), 2, 1)
    screen.blit(screen2, (shake[0], shake[1]))
    screen.blit(screenT, (shake[0], shake[1]))
    screen.blit(screenUI, (0, 0))
    pygame.display.update()
    clock.tick(fps)
