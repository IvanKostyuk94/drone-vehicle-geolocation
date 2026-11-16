"""Full-frame predictions of the split evaluation.split
({split}_frame_predictions.pt)."""

import json

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dronegeo.config import load_config
from dronegeo.data import load_coco
from dronegeo.infer import predict_frame
from dronegeo.model import build_model
from dronegeo.train import get_device


def evaluate_frames(cfg: dict) -> None:
    """Run predict_frame on every frame of the split and save predictions
    and ground truth to runs/<run>/{split}_frame_predictions.pt."""
    device = get_device()
    images, boxes = load_coco(cfg["paths"]["coco_annotations"])
    matches = pd.read_csv(cfg["paths"]["matches_csv"])
    images = images.merge(
        matches, left_on="file_name", right_on="image", how="left"
    )
    with open(cfg["paths"]["splits"]) as f:
        splits = json.load(f)
    val_images = images.loc[
        images["best"].isin(splits[cfg["evaluation"]["split"]])
    ]
    model = build_model(pretrained=False)
    model.to(device)
    model.eval()
    run_dir = cfg["paths"]["runs"] / cfg["evaluation"]["run"]
    state = torch.load(run_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(state["model"])
    image_dir = cfg["paths"]["images"]
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for _, image in tqdm(
            val_images.iterrows(), desc="predicting", total=len(val_images)
        ):
            file_name = image["file_name"]
            image_id = image["image_id"]
            gt_boxes = boxes.loc[boxes["image_id"] == image_id]
            gt_boxes = {
                "boxes": torch.tensor(
                    gt_boxes[["x1", "y1", "x2", "y2"]].values,
                    dtype=torch.float32,
                ),
                "labels": torch.tensor(
                    gt_boxes["category_id"].values, dtype=torch.int64
                ),
            }
            img = cv2.imread(str(image_dir / file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img / 255).astype(np.float32)
            img = torch.from_numpy(img).permute(2, 0, 1)
            result = predict_frame(
                model,
                img,
                size=cfg["tiling"]["size"],
                overlap=cfg["tiling"]["overlap"],
                device=device,
                iou_threshold=cfg["inference"]["nms_iou_threshold"],
            )
            predictions.append(result)
            ground_truth.append(gt_boxes)
    torch.save(
        {"predictions": predictions, "ground_truth": ground_truth},
        run_dir / f"{cfg['evaluation']['split']}_frame_predictions.pt",
    )


if __name__ == "__main__":
    cfg = load_config()
    evaluate_frames(cfg)
