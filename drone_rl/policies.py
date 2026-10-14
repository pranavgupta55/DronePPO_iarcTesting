"""Loading a run's policy for evaluation and the viewer. A policy is anything with .reset(env) and .act(env, obs) -> action (4 numbers).
Trained runs are numpy forward passes of their saved actor; 'controller' is the hand-written bang-bang controller."""
import json
from pathlib import Path

import numpy as np

from drone_rl.env import NumpyPolicy, NORM_CLIP, make_spec

CONTROLLER = "controller"


class NetworkPolicy:
    """The saved actor's mean action (no exploration noise) on normalized observations."""

    def __init__(self, layers, obs_mean, obs_std):
        self.layers, self.obs_mean, self.obs_std = layers, obs_mean, obs_std
        self._net = NumpyPolicy(layers, np.zeros(4))

    def reset(self, env):
        pass

    def act(self, env, obs):
        return self._net.mean(np.clip((obs - self.obs_mean) / self.obs_std, -NORM_CLIP, NORM_CLIP))


class ControllerPolicy:
    def __init__(self, params=None):
        from drone_rl.controller import DEFAULTS
        self.params = dict(DEFAULTS, **(params or {}))

    def reset(self, env):
        for key in ("bb_braking", "bb_unwound"):
            if hasattr(env, key):
                delattr(env, key)

    def act(self, env, obs):
        from drone_rl.controller import control
        return control(env, self.params)


def load_layers(run_dir):
    """[(weight, bias), ...], observation mean and std of a saved run."""
    import torch  # only needed here, so worker processes that never load a policy skip it
    run_dir = Path(run_dir)
    state = torch.load(run_dir / "actor.pt", map_location="cpu")
    keys = sorted({int(k.split(".")[1]) for k in state if k.startswith("net.")})
    layers = [(state[f"net.{i}.weight"].numpy().astype(np.float64), state[f"net.{i}.bias"].numpy().astype(np.float64)) for i in keys]
    norms = np.load(run_dir / "norm.npz")
    return layers, norms["obs_mean"], np.sqrt(norms["obs_var"] + 1e-8)


def load_policy(run_dir):
    """The policy of a run folder (a trained network, or the controller for a run with kind 'controller')."""
    run_dir = Path(run_dir)
    spec = make_spec(json.loads((run_dir / "spec.json").read_text()))
    if spec["kind"] == CONTROLLER:
        return ControllerPolicy(spec["params"].get("controller"))
    layers, obs_mean, obs_std = load_layers(run_dir)
    if not spec["obs_norm"]:
        obs_mean, obs_std = np.zeros_like(obs_mean), np.ones_like(obs_std)
    return NetworkPolicy(layers, obs_mean, obs_std)


def network_layers(run_dir):
    """Layers for drawing the network in the viewer; None for the controller."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    if spec.get("kind") == CONTROLLER:
        return None
    return load_layers(run_dir)[0]
