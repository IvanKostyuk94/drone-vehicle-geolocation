"""Split the intersections into train, validation and test (splits.json)."""

import json

import numpy as np
import pandas as pd

from dronegeo.config import load_config
from dronegeo.data import load_coco, split_set

cfg = load_config()
images, boxes = load_coco(cfg["paths"]["coco_annotations"])
matches = pd.read_csv(cfg["paths"]["matches_csv"])

df = boxes.merge(images, on="image_id").merge(
    matches, left_on="file_name", right_on="image"
)
train, validate, test = split_set(df, np.random.default_rng(42))

with open(cfg["paths"]["splits"], "w") as f:
    json.dump({"train": train, "val": validate, "test": test}, f, indent=2)
