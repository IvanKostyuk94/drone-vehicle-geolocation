"""Train the detector with the settings in configs/config.yaml."""

from dronegeo.config import load_config
from dronegeo.train import fit

if __name__ == "__main__":
    cfg = load_config()
    fit(cfg)
