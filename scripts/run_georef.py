"""Georeferenced detections of the split evaluation.split
({split}_detections.geojson)."""

import json
import os

import pandas as pd
import torch
from tqdm import tqdm

from dronegeo.config import load_config
from dronegeo.data import load_coco
from dronegeo.georef import (
    frame_to_lonlat,
    frame_to_master,
    homography_ok,
    read_master_homography,
    read_ortho_params,
)


def georeference_frames(cfg: dict) -> None:
    """Convert the detection centres (score >= evaluation.score_threshold)
    of every frame to longitude and latitude.

    Writes runs/<run>/{split}_detections.geojson and prints the number of
    frames skipped by homography_ok.
    """
    images, _ = load_coco(cfg["paths"]["coco_annotations"])
    matches = pd.read_csv(cfg["paths"]["matches_csv"])
    images = images.merge(
        matches, left_on="file_name", right_on="image", how="left"
    )
    detection_path = os.path.join(
        cfg["paths"]["runs"],
        cfg["evaluation"]["run"],
        f"{cfg['evaluation']['split']}_frame_predictions.pt",
    )
    detections = torch.load(detection_path)
    with open(cfg["paths"]["splits"]) as f:
        splits = json.load(f)
    parameters = {}
    for letter in splits[cfg["evaluation"]["split"]]:
        parameters[letter] = read_ortho_params(
            cfg["paths"]["orthophotos"],
            letter,
            cutout_original_size=cfg["georef"]["cutout_original_size"],
        )

    val_images = images.loc[
        images["best"].isin(splits[cfg["evaluation"]["split"]])
    ]
    counter = 0
    features = []
    for (_, image), pred in tqdm(
        zip(val_images.iterrows(), detections["predictions"]),
        desc="georeferencing",
        total=len(val_images),
    ):
        img_path = os.path.join(cfg["paths"]["images"], image["file_name"])
        master_path = os.path.join(
            cfg["paths"]["master_frames"], f"{image['best']}.png"
        )
        frame_master, n_inliers = frame_to_master(img_path, master_path)
        ok = homography_ok(
            frame_master,
            n_inliers,
            cfg["georef"]["min_inliers"],
            cfg["georef"]["scale_range"],
        )
        if not ok:
            counter += 1
            continue
        relevant = pred["scores"] >= cfg["evaluation"]["score_threshold"]
        rel_scores = pred["scores"][relevant]
        rel_labels = pred["labels"][relevant]
        rel_boxes = pred["boxes"][relevant]
        centers_x = (rel_boxes[:, 0] + rel_boxes[:, 2]) / 2.0
        centers_y = (rel_boxes[:, 1] + rel_boxes[:, 3]) / 2.0
        centers = torch.stack([centers_x, centers_y], dim=1)
        path_master_homography = os.path.join(
            cfg["paths"]["master_frames"], f"{image['best']}.txt"
        )
        master_trans = read_master_homography(path_master_homography)

        coord = frame_to_lonlat(
            centers.numpy(),
            frame_master,
            master_trans,
            parameters[image["best"]],
        )
        for (lon, lat), label, score in zip(coord, rel_labels, rel_scores):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "frame": image["file_name"],
                        "intersection": image["best"],
                        "class": int(label),
                        "score": float(score),
                    },
                }
            )
    save_path = os.path.join(
        cfg["paths"]["runs"],
        cfg["evaluation"]["run"],
        f"{cfg['evaluation']['split']}_detections.geojson",
    )
    with open(save_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"{counter} frames were skipped")


if __name__ == "__main__":
    cfg = load_config()
    georeference_frames(cfg)
