from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_config(path: Path = PROJECT_DIR / "configs" / "config.yaml") -> dict:
    """Read the YAML config.

    Returns the config as a dict, with every entry of "paths" as an absolute
    Path.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {name: PROJECT_DIR / p for name, p in cfg["paths"].items()}
    return cfg
