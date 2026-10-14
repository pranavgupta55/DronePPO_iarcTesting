"""The live network view: the traced forward pass matches the policy's action, and the live drawing works on a headless surface."""
import os

import numpy as np
import pytest

from drone_rl.policies import NetworkPolicy


def make_policy(seed=0):
    rng = np.random.default_rng(seed)
    layers = [(rng.normal(0, 0.5, (16, 24)), rng.normal(0, 0.1, 16)), (rng.normal(0, 0.5, (8, 16)), rng.normal(0, 0.1, 8)), (rng.normal(0, 0.5, (4, 8)), rng.normal(0, 0.1, 4))]
    return NetworkPolicy(layers, rng.normal(0, 1, 24), np.abs(rng.normal(1, 0.2, 24)))


def test_trace_ends_in_the_action_and_has_every_layer():
    policy = make_policy()
    obs = np.random.default_rng(1).normal(0, 1, 24)
    values = policy.trace(obs)
    assert [len(v) for v in values] == [24, 16, 8, 4]
    assert np.allclose(values[-1], policy.act(None, obs))
    assert np.all(np.abs(values[1]) <= 1) and np.all(np.abs(values[2]) <= 1)  # hidden layers are tanh


def test_live_drawing_runs_and_marks_pixels():
    pygame = pytest.importorskip("pygame")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from drone_rl.viewer.network_draw import draw_live_network, live_layout
    pygame.init()
    policy = make_policy()
    obs = np.random.default_rng(2).normal(0, 1, 24)
    width, height, _, _ = live_layout([24, 16, 8, 4], 32, 5.5)
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    draw_live_network(surface, policy.layers, policy.trace(obs), 32, 5.5)
    assert pygame.mask.from_surface(surface, 10).count() > 500  # lines and nodes were drawn
    before = pygame.image.tostring(surface, "RGBA")
    surface.fill((0, 0, 0, 0))
    draw_live_network(surface, policy.layers, policy.trace(obs * 2.0), 32, 5.5)
    assert pygame.image.tostring(surface, "RGBA") != before  # a different input draws a different picture
