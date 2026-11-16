"""Annotations, tiling, dataset and augmentation.

Targets follow the torchvision detection format: dict with
"boxes" float32 [K, 4] (x1, y1, x2, y2) in tile pixels and
"labels" int64 [K] (1 car, 2 bus, 3 truck, 4 motorcycle).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_coco(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a COCO annotation file.

    Returns (images, boxes): images with columns image_id, file_name, width,
    height; boxes with columns image_id, category_id, x1, y1, x2, y2, w, h.
    """
    with open(path) as f:
        coco = json.load(f)

    images = pd.DataFrame(coco["images"]).rename(columns={"id": "image_id"})
    images = images[["image_id", "file_name", "width", "height"]]

    boxes = pd.DataFrame(coco["annotations"])
    boxes[["x1", "y1", "w", "h"]] = pd.DataFrame(
        boxes["bbox"].tolist(), index=boxes.index
    )
    boxes["x2"] = boxes["x1"] + boxes["w"]
    boxes["y2"] = boxes["y1"] + boxes["h"]
    boxes = boxes[
        ["image_id", "category_id", "x1", "y1", "x2", "y2", "w", "h"]
    ]
    return images, boxes


def split_set(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[list, list, list]:
    """Split the intersections into about 70/15/15 % of the frames.

    df: boxes merged with images and matches.csv (columns best, file_name).
    Returns (train, validate, test) lists of intersection letters.
    """
    frames = df.groupby("best").file_name.nunique()
    ser = frames.sample(frac=1, random_state=rng)
    tot = ser.sum()
    tot_count = 0
    train = []
    validate = []
    test = []

    for gr, count in ser.items():
        tot_count += count
        if tot_count < 0.7 * tot:
            train.append(gr)
        elif 0.7 * tot <= tot_count <= 0.85 * tot:
            validate.append(gr)
        else:
            test.append(gr)
    return (train, validate, test)


def test_size(
    box: pd.Series,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    threshold: float,
) -> bool:
    """True if the clipped box (x1, x2, y1, y2) keeps more than threshold of
    the area of the original box."""
    ar = (x2 - x1) * (y2 - y1)
    ar_box = box.w * box.h
    return ar > threshold * ar_box


def get_in_boxes(
    start_u: int,
    end_u: int,
    start_v: int,
    end_v: int,
    boxes: pd.DataFrame,
    threshold: float = 0.3,
) -> dict:
    """Boxes of one frame inside the tile [start_u:end_u, start_v:end_v].

    boxes: rows of the load_coco box table for that frame.
    Returns dict(boxes float32 [K, 4] in tile pixels, labels int64 [K]).
    """
    contained_boxes = {"boxes": [], "labels": []}
    for _, box in boxes.iterrows():
        if (
            (box.x2 < start_v)
            or (box.x1 > end_v)
            or (box.y2 < start_u)
            or (box.y1 > end_u)
        ):
            continue
        elif (
            (box.x2 < end_v)
            and (box.x1 > start_v)
            and (box.y2 < end_u)
            and (box.y1 > start_u)
        ):
            contained_boxes["boxes"].append(
                [
                    box.x1 - start_v,
                    box.y1 - start_u,
                    box.x2 - start_v,
                    box.y2 - start_u,
                ]
            )
            contained_boxes["labels"].append(box.category_id)
        else:
            new_x1 = max(start_v, box.x1) - start_v
            new_x2 = min(end_v, box.x2) - start_v
            new_y1 = max(start_u, box.y1) - start_u
            new_y2 = min(end_u, box.y2) - start_u
            if test_size(box, new_x1, new_x2, new_y1, new_y2, threshold):
                contained_boxes["boxes"].append(
                    [new_x1, new_y1, new_x2, new_y2]
                )
                contained_boxes["labels"].append(box.category_id)
    contained_boxes["boxes"] = np.array(
        contained_boxes["boxes"], dtype=np.float32
    ).reshape(-1, 4)
    contained_boxes["labels"] = np.array(
        contained_boxes["labels"], dtype=np.int64
    )
    return contained_boxes


def get_tiles(
    img_height: int, img_width: int, size: int, overlap: int
) -> dict:
    """Top-left corners of the tiles covering one frame.

    Returns dict(start_u: row indices, start_v: column indices), one entry
    per tile.
    """
    stride = size - overlap
    start_last_u = img_height - size
    start_last_v = img_width - size
    tiles = {"start_u": [], "start_v": []}
    start_u = 0
    start_v = 0
    n_u = int(np.ceil((img_height - size) / stride)) + 1
    n_v = int(np.ceil((img_width - size) / stride)) + 1
    for _ in range(n_u):
        if start_u > start_last_u:
            start_u = start_last_u
        for _ in range(n_v):
            if start_v > start_last_v:
                start_v = start_last_v
            tiles["start_u"].append(start_u)
            tiles["start_v"].append(start_v)
            start_v += stride
        start_u += stride
        start_v = 0
    return tiles


def get_tile_table(
    images: pd.DataFrame, size: int, overlap: int
) -> pd.DataFrame:
    """One row per tile of every image.

    Returns a DataFrame with columns file_name, image_id, start_u, start_v.
    """
    df_dict = {"file_name": [], "image_id": [], "start_u": [], "start_v": []}
    for row in images.itertuples():
        tiles = get_tiles(row.height, row.width, size, overlap)
        df_dict["file_name"].extend([row.file_name] * len(tiles["start_u"]))
        df_dict["image_id"].extend([row.image_id] * len(tiles["start_u"]))
        df_dict["start_u"].extend(tiles["start_u"])
        df_dict["start_v"].extend(tiles["start_v"])
    df = pd.DataFrame(df_dict)
    return df


class TileDataset(Dataset):
    """Tiles as (image, target) pairs.

    image: float32 tensor [3, size, size] in [0, 1]; target: dict(boxes,
    labels). Random flips and rotations if train is True.
    """

    def __init__(
        self,
        tiles: pd.DataFrame,
        boxes: pd.DataFrame,
        image_dir: Path,
        train: bool,
        size: int,
        min_visible: float,
    ):
        self.tiles = tiles
        self.boxes = boxes
        self.image_dir = image_dir
        self.train = train
        self.size = size
        self.min_visible = min_visible

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        tile = self.tiles.iloc[idx]
        image_id = tile.image_id
        img = cv2.imread(str(self.image_dir / tile.file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tile_img = img[
            tile.start_u : tile.start_u + self.size,
            tile.start_v : tile.start_v + self.size,
            :,
        ]
        frame_boxes = self.boxes[self.boxes.image_id == image_id]
        target = get_in_boxes(
            start_u=tile.start_u,
            end_u=tile.start_u + self.size,
            start_v=tile.start_v,
            end_v=tile.start_v + self.size,
            boxes=frame_boxes,
            threshold=self.min_visible,
        )
        tile_img = (tile_img / 255).astype(np.float32)
        tensor = torch.from_numpy(tile_img).permute(2, 0, 1)
        target["boxes"] = torch.from_numpy(target["boxes"])
        target["labels"] = torch.from_numpy(target["labels"])
        if self.train:
            tensor, target["boxes"] = random_flip_rotate(
                tensor, target["boxes"]
            )
        return tensor, target


def collate_fn(batch: list) -> tuple[list, list]:
    """List of (image, target) pairs -> (list of images, list of targets)."""
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets


def random_flip_rotate(
    image: torch.Tensor, boxes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random rotation by a multiple of 90 degrees, optionally mirrored.

    image: [3, H, W], boxes: [K, 4] (x1, y1, x2, y2).
    Returns the transformed (image, boxes).
    """
    R_90 = torch.tensor([[0.0, -1.0], [1, 0.0]], dtype=torch.float32)
    R_180 = torch.tensor([[-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)
    R_270 = torch.tensor([[0.0, 1.0], [-1, 0.0]], dtype=torch.float32)
    R_x_mir = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    R_90_mir = R_90 @ R_x_mir
    R_180_mir = R_180 @ R_x_mir
    R_270_mir = R_270 @ R_x_mir
    transformations = {
        1: R_90,
        2: R_180,
        3: R_270,
        4: R_x_mir,
        5: R_90_mir,
        6: R_180_mir,
        7: R_270_mir,
    }
    center = torch.tensor([image.shape[1] / 2, image.shape[2] / 2])
    throw = torch.randint(8, (1,)).item()

    if throw == 0:
        return image, boxes
    else:
        trans = transformations[throw]

        trans_boxes = (
            (boxes.reshape(-1, 2) - center) @ trans + center
        ).reshape(-1, 2, 2)
        top_left = trans_boxes.min(dim=1).values
        bottom_right = trans_boxes.max(dim=1).values
        trans_boxes = torch.cat([top_left, bottom_right], dim=1)
        flip = throw // 4
        k = throw % 4
        trans_image = torch.rot90(image, k=k, dims=(1, 2))
        if flip:
            trans_image = torch.flip(trans_image, dims=(2,))
        return trans_image, trans_boxes
