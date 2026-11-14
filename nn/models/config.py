"""Per-model architecture configs (models/configs/<name>.yaml) and a model builder.

Each config holds the constructor arguments of one model class (except in_channels,
which follows DETECTORS) plus its BCEWithLogitsLoss pos_weight. Training settings
(epochs, learning rate, ...) stay in globals.py.
"""

import importlib
import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
MODEL_NAMES = sorted(
    f[: -len(".yaml")] for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml")
)


def load_config(name):
    with open(os.path.join(CONFIG_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f) or {}


def build_model(name, **overrides):
    """Build `name` from its config. Returns (model, pos_weight).

    overrides replace config values, e.g. out_channels=16 or input_len=19456.
    Imports models.<name>, so run from nn/.
    """
    from models.globals import DETECTORS

    kwargs = {**load_config(name), **overrides}
    pos_weight = kwargs.pop("pos_weight")
    if not name.startswith("GregNet"):
        kwargs["in_channels"] = len(DETECTORS)
    model_class = getattr(importlib.import_module(f"models.{name}"), name)
    return model_class(**kwargs), pos_weight
