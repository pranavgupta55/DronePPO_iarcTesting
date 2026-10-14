"""The viewer: python3 -m drone_rl view [run id or name]   (or run main_screen.py in the repo root).
It flies a saved run toward the mouse. With no argument it opens the newest finished run. Left / Right step to the previous / next run in the current list,
Tab opens the run menu (menu.py), Space hides the HUD, Esc quits."""
import math
import sys
import time

import numpy as np
import pygame

from drone_rl import registry
from drone_rl.config import THRUSTER_ANGLE_LIMITS, TIME_STEP_DELTA, REACH_DIST
from drone_rl.env import DroneEnv, load_scheme, make_spec
from drone_rl.policies import load_policy, network_layers
from drone_rl.viewer.fontDict import fonts
from drone_rl.viewer.menu import MenuView
from drone_rl.viewer.network_draw import NeuralNetwork, draw_network, tanh, tanh_prime
from drone_rl.viewer.text import drawText, wrapText

FPS = 60
TARGET_MARGIN = 10
EPISODE_SECONDS = 500
GAUGE_OUTER, GAUGE_INNER, GAUGE_GAP, GAUGE_MARGIN = 56, 18, 20, 12  # thruster gauge: outer / inner radius, gap between the two sectors, distance from the screen's corner (px)
LABEL_WIDTH = 460


class Endesga:
    my_blue = [32, 36, 46]
    black = [19, 19, 19]
    white = [255, 255, 255]
    greyL = [200, 200, 200]
    grey = [150, 150, 150]
    greyD = [100, 100, 100]
    greyVD = [50, 50, 50]
    cream = [237, 171, 80]
    debug_red = [255, 96, 141]
    network_green = [64, 128, 67]
    network_red = [127, 45, 41]
    sebastian_lague_light_purple = [137, 133, 181]


def layers_to_network(layers):
    """Copies (weight, bias) layers into a NeuralNetwork so draw_network can draw it."""
    network = NeuralNetwork([layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers], tanh, tanh_prime)
    for dense, (weight, bias) in zip((layer for layer in network.layers if hasattr(layer, "weights")), layers):
        dense.weights = np.asarray(weight, dtype=float)
        dense.bias = np.asarray(bias, dtype=float).reshape(-1, 1)
    return network


def draw_thruster_gauge(surface, drone, screen_width):
    """Top right: one downward-facing annulus sector per thruster covering its swivel range (left thruster on the left), with a red line along the thruster's
    current direction relative to the drone body; the line runs from the inner arc out to a radius proportional to the thruster's force (the outer arc is full force)."""
    right_center_x = screen_width - GAUGE_MARGIN - GAUGE_OUTER
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


def draw_drone(surface, drone, scale):
    """The drone with positions scaled into the zoomed-out view (line widths and marker sizes stay in screen pixels)."""
    points = [(float(point[0]) * scale, float(point[1]) * scale) for point in drone.shell.points]
    pygame.draw.line(surface, Endesga.white, points[0], points[1], 2)
    for point, angle in zip(points, drone.thruster_angles):
        absolute = angle + drone.shell.angle
        pygame.draw.circle(surface, Endesga.debug_red, point, 3)
        pygame.draw.line(surface, Endesga.debug_red, point, (point[0] + 10 * math.cos(absolute), point[1] + 10 * math.sin(absolute)), 2)


class Session:
    """One loaded run: its environment (in the run's own world), policy, labels and the drawn network."""

    def __init__(self, run_dir, meta):
        import json
        self.run_dir, self.meta = run_dir, meta
        spec = make_spec(json.loads((run_dir / "spec.json").read_text()))
        module, scheme = load_scheme(spec["scheme"], spec["params"])
        self.env = DroneEnv(spec, module, scheme, max_steps=int(EPISODE_SECONDS / TIME_STEP_DELTA))
        self.env.set_ramp(1.0)
        self.env.set_eval_mode()
        self.policy = load_policy(run_dir)
        self.policy.reset(self.env)
        self.world = (self.env.W, self.env.H)
        self.title = f"{meta['name']}: {meta['title']}" if meta.get("title") and meta["title"] != meta["name"] else meta["name"]
        self.blurb = meta.get("blurb_text", "")
        layers = network_layers(run_dir)
        self.network_surface, self.network_corner = None, (0, 0)
        if layers is not None:
            network = layers_to_network(layers)
            dense = [layer for layer in network.layers if hasattr(layer, "weights")]
            tallest = max(max(layer.weights.shape) for layer in dense)
            self.network_surface = pygame.Surface((20 + 60 * len(dense), 200), pygame.SRCALPHA)
            draw_network(network, self.network_surface, (10, 100), Endesga.network_green, Endesga.network_red, spacing_x=60, spacing_y=min(3, 180 / tallest))
            bounds = self.network_surface.get_bounding_rect()
            self.network_corner = (GAUGE_MARGIN - bounds.x, GAUGE_MARGIN - bounds.y)
        self.env.reset()
        self.policy.reset(self.env)


def flyable_rows():
    return [r for r in registry.load_index() if r.get("flyable") and not r.get("archived") and r.get("status") == "ok"]


def open_session(row_id):
    run_dir = registry.RUNS / row_id
    return Session(run_dir, registry.read_meta(run_dir))


def main():
    rows = flyable_rows()
    if not rows:
        sys.exit("no flyable runs in runs/ (train one with python3 -m drone_rl batch ...)")
    order = [r["id"] for r in rows]  # newest first
    current = order[0]
    if len(sys.argv) > 1:
        current = registry.find_run(sys.argv[1]).name
    pygame.init()
    screen = pygame.display.set_mode((1512, 982), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    screen_width, screen_height = screen.get_size()
    surface_ui = pygame.Surface((screen_width, screen_height)).convert_alpha()
    session = open_session(current)
    view_scale, world, world_offset = 1.0, None, (0, 0)

    def layout():
        nonlocal view_scale, world, world_offset
        W, H = session.world
        view_scale = min(screen_width / W, screen_height / H, 1.0)  # shrink the world to fit the screen, never enlarge it
        size = (int(W * view_scale), int(H * view_scale))
        world = pygame.Surface(size)  # drawn at screen size directly, so the frame rate doesn't depend on the world size
        world_offset = ((screen_width - size[0]) // 2, (screen_height - size[1]) // 2)

    layout()
    hud, accumulator, last_time, running = True, 0.0, time.time(), True
    while running:
        mx, my = pygame.mouse.get_pos()
        now = time.time()
        accumulator = min(accumulator + now - last_time, 0.25)
        last_time = now
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    hud = not hud
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    step = -1 if event.key == pygame.K_LEFT else 1
                    current = order[(order.index(current) + step) % len(order)] if current in order else order[0]
                    session = open_session(current)
                    layout()
                    accumulator = 0.0
        env = session.env
        mouse_target = np.array([(mx - world_offset[0]) / view_scale, (my - world_offset[1]) / view_scale])
        mouse_target = np.clip(mouse_target, TARGET_MARGIN, [env.W - TARGET_MARGIN, env.H - TARGET_MARGIN])  # a target in or beyond a wall is outside what a run was built for
        env.target = mouse_target
        obs = env.observe()
        while accumulator >= TIME_STEP_DELTA:
            obs, _, terminated, truncated = env.step(session.policy.act(env, obs))
            if terminated or truncated:
                env.reset()
                session.policy.reset(env)
            env.target = mouse_target  # a reach or a reset spawns a random target; the mouse overrides it
            obs = env.observe()
            accumulator -= TIME_STEP_DELTA

        screen.fill(Endesga.my_blue)
        surface_ui.fill((0, 0, 0, 0))
        world.fill(Endesga.my_blue)
        if hud:
            pygame.draw.rect(world, Endesga.black, world.get_rect(), 1)
            pygame.draw.circle(world, Endesga.cream, env.target * view_scale, max(REACH_DIST * view_scale, 3), 1)
        draw_drone(world, env.drone, view_scale)
        screen.blit(world, world_offset)
        if hud:
            drawText(surface_ui, Endesga.sebastian_lague_light_purple, fonts["regular16"], 5, screen_height - 24, f"{round(clock.get_fps())} fps", Endesga.black, 0)
            if session.network_surface is not None:
                surface_ui.blit(session.network_surface, session.network_corner)
            draw_thruster_gauge(surface_ui, env.drone, screen_width)
            label_font, blurb_font = fonts["bold20"], fonts["regular16"]
            blurb_lines, _ = wrapText(blurb_font, session.blurb, LABEL_WIDTH)
            y = screen_height - 12 - len(blurb_lines) * blurb_font.get_height() * 1.1 - label_font.get_height() * 1.2
            drawText(surface_ui, Endesga.white, label_font, screen_width - 12, y, session.title, Endesga.black, 1, justify="right")
            drawText(surface_ui, Endesga.greyL, blurb_font, screen_width - 12, y + label_font.get_height() * 1.2, session.blurb, Endesga.black, 1, wrap=True, maxLen=LABEL_WIDTH, justify="right")
        pygame.mouse.set_visible(False)
        pygame.draw.circle(surface_ui, Endesga.black, (mx + 1, my + 1), 2, 1)
        pygame.draw.circle(surface_ui, Endesga.white, (mx, my), 2, 1)
        screen.blit(surface_ui, (0, 0))
        pygame.display.update()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    main()
