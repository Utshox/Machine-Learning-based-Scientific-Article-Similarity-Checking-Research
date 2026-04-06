import json
import os


DEFAULT_CONFIG = {
    "semantic_weight": 0.7,
    "structural_weight": 0.3,
    "threshold": 0.45,
    "window_size": 150,
    "step_size": 25,
}


def get_config_path():
    return os.path.join(os.path.dirname(__file__), "trained_config.json")


def load_config():
    path = get_config_path()
    if not os.path.exists(path):
        return DEFAULT_CONFIG.copy()

    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    config = DEFAULT_CONFIG.copy()
    config.update(loaded)
    return config


def save_config(config):
    path = get_config_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return path
