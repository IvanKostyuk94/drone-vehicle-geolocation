"""Training and validation loops."""

import json
import math
import os
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dronegeo.data import TileDataset, collate_fn, get_tile_table, load_coco
from dronegeo.model import build_model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    grad_clip_norm: float,
    on_step: Callable[[int, dict], None] | None = None,
    first_step: int = 0,
    skip: int = 0,
) -> dict:
    """One pass over the training data.

    on_step(step, loss_rec) is called after every batch; the first skip
    batches are passed over (resuming).
    Returns a dict of per-batch lists: the four losses and learning_rate.
    """
    model.train()
    loss_rec = None
    for i, (images, targets) in enumerate(tqdm(loader, desc="training")):
        if i < skip:
            continue
        images = [img.to(device) for img in images]
        targets = [
            {key: value.to(device) for key, value in t.items()}
            for t in targets
        ]
        loss = model(images, targets)
        tot_loss = sum(loss.values())
        optimizer.zero_grad()
        if torch.isfinite(tot_loss):
            tot_loss.backward()
        else:
            raise ValueError("Loss is not finite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        if loss_rec is None:
            loss_rec = {key: [value.item()] for key, value in loss.items()}
            loss_rec["learning_rate"] = [scheduler.get_last_lr()[0]]
        else:
            for key, value in loss.items():
                loss_rec[key].append(value.item())
            loss_rec["learning_rate"].append(scheduler.get_last_lr()[0])
        scheduler.step()
        step = first_step + i + 1
        if on_step is not None:
            on_step(step, loss_rec)
    return loss_rec


def evaluate_loss(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict:
    """Mean of each loss over loader, without gradients."""
    model.train()
    loss_rec = None
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [
                {key: value.to(device) for key, value in t.items()}
                for t in targets
            ]
            loss = model(images, targets)
            tot_loss = sum(loss.values())
            if not torch.isfinite(tot_loss):
                raise ValueError("Loss is not finite")

            if loss_rec is None:
                loss_rec = {key: [value.item()] for key, value in loss.items()}
            else:
                for key, value in loss.items():
                    loss_rec[key].append(value.item())
    loss_avg = {key: np.mean(values) for key, values in loss_rec.items()}
    return loss_avg


def get_device() -> torch.device:
    """cuda if available, else mps, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def factor(step: int, cfg: dict, train_loader: DataLoader) -> float:
    """Learning rate factor: linear warm-up, then cosine decay to 0."""
    warmup = cfg["training"]["warmup"]
    total = cfg["training"]["epochs"] * len(train_loader)
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def fit(cfg: dict) -> None:
    """Full training run.

    Writes training logs, validation logs and checkpoint.pt to
    runs/<date-time>/, or continues the run named in training.resume.
    """

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
    train_tiles = tiles.loc[tiles["best"].isin(splits["train"])]
    if cfg["training"]["max_train_tiles"] is not None:
        train_tiles = train_tiles.sample(
            cfg["training"]["max_train_tiles"], random_state=42
        )
    val_tiles = tiles.loc[tiles["best"].isin(splits["val"])]
    val_tiles = val_tiles.sample(500, random_state=42)
    train_set = TileDataset(
        tiles=train_tiles,
        boxes=boxes,
        image_dir=cfg["paths"]["images"],
        train=True,
        size=cfg["tiling"]["size"],
        min_visible=cfg["tiling"]["min_visible"],
    )
    val_set = TileDataset(
        tiles=val_tiles,
        boxes=boxes,
        image_dir=cfg["paths"]["images"],
        train=False,
        size=cfg["tiling"]["size"],
        min_visible=cfg["tiling"]["min_visible"],
    )

    gen = torch.Generator()
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        generator=gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model()
    model.to(device)
    optimizer = torch.optim.SGD(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: factor(step, cfg, train_loader)
    )

    if cfg["training"]["resume"] is None:
        run_dir = cfg["paths"]["runs"] / datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        run_dir.mkdir(parents=True)
        start_step = 0
        old_log = None
    else:
        run_dir = cfg["paths"]["runs"] / cfg["training"]["resume"]
        state = torch.load(run_dir / "checkpoint.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        log_path = (
            run_dir / f"train_log_epoch{start_step // len(train_loader)}.csv"
        )
        if log_path.exists():
            old_log = pd.read_csv(log_path)
            old_log = old_log[old_log["step"] <= start_step]
        else:
            old_log = None

    def on_step(step: int, loss_rec: dict) -> None:
        if step % cfg["training"]["eval_every"] == 0:
            epoch = (step - 1) // len(train_loader)
            df = pd.DataFrame(loss_rec)
            df["step"] = range(step - len(df) + 1, step + 1)
            if old_log is not None and epoch == start_step // len(
                train_loader
            ):
                df = pd.concat([old_log, df])
            df.to_csv(run_dir / f"train_log_epoch{epoch}.csv", index=False)

            val_path = run_dir / "val_log.csv"
            row = evaluate_loss(model, val_loader, device)
            row["step"] = step
            pd.DataFrame([row]).to_csv(
                val_path, mode="a", header=not val_path.exists(), index=False
            )
            tmp_path = run_dir / "checkpoint.pt.tmp"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                },
                tmp_path,
            )
            os.replace(tmp_path, run_dir / "checkpoint.pt")

    n_batches = len(train_loader)
    start_epoch = start_step // n_batches
    for i in range(start_epoch, cfg["training"]["epochs"]):
        gen.manual_seed(42 + i)
        first_step = i * n_batches
        skip = max(0, start_step - first_step)
        loss_rec = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_clip_norm=cfg["training"]["grad_clip_norm"],
            on_step=on_step,
            first_step=first_step,
            skip=skip,
        )
        df = pd.DataFrame(loss_rec)
        step = (i + 1) * len(train_loader)
        df["step"] = range(step - len(df) + 1, step + 1)
        if old_log is not None and i == start_step // len(train_loader):
            df = pd.concat([old_log, df])

        df.to_csv(run_dir / f"train_log_epoch{i}.csv", index=False)
        tmp_path = run_dir / "checkpoint.pt.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            tmp_path,
        )
        os.replace(tmp_path, run_dir / "checkpoint.pt")
