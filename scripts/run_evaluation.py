"""Tile predictions of the split evaluation.split ({split}_predictions.pt)."""

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dronegeo.config import load_config
from dronegeo.data import TileDataset, collate_fn, get_tile_table, load_coco
from dronegeo.evaluate import predict
from dronegeo.model import build_model
from dronegeo.train import get_device


def val(cfg: dict) -> None:
    """Predict all tiles of the split and save predictions and ground truth
    to runs/<run>/{split}_predictions.pt."""
    device = get_device()
    images, boxes = load_coco(cfg["paths"]["coco_annotations"])
    tiles = get_tile_table(
        images, cfg["tiling"]["size"], cfg["tiling"]["overlap"]
    )
    matches = pd.read_csv(cfg["paths"]["matches_csv"])
    tiles = tiles.merge(
        matches, left_on="file_name", right_on="image", how="left"
    )
    with open(cfg["paths"]["splits"]) as f:
        splits = json.load(f)
    val_tiles = tiles.loc[
        tiles["best"].isin(splits[cfg["evaluation"]["split"]])
    ]
    val_set = TileDataset(
        tiles=val_tiles,
        boxes=boxes,
        image_dir=cfg["paths"]["images"],
        train=False,
        size=cfg["tiling"]["size"],
        min_visible=cfg["tiling"]["min_visible"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = build_model(pretrained=False)
    model.to(device)
    run_dir = cfg["paths"]["runs"] / cfg["evaluation"]["run"]
    state = torch.load(run_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(state["model"])
    predictions, ground_truth = predict(model, val_loader, device)
    torch.save(
        {"predictions": predictions, "ground_truth": ground_truth},
        run_dir / f"{cfg['evaluation']['split']}_predictions.pt",
    )


if __name__ == "__main__":
    cfg = load_config()
    val(cfg)
