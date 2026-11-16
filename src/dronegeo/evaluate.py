import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dronegeo.boxes import box_iou


def match_tile(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Greedy matching of detections to ground truth (one tile, one class).

    pred_boxes: [D, 4], pred_scores: [D], gt_boxes: [G, 4].
    Returns bool [D], True for true positives, in the original order.
    """
    order = torch.argsort(pred_scores, descending=True)
    pred_boxes_sorted = pred_boxes[order]
    used = torch.zeros(
        gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device
    )
    detection = torch.zeros(
        pred_boxes.shape[0], dtype=torch.bool, device=pred_boxes.device
    )
    iou = box_iou(pred_boxes_sorted, gt_boxes)
    for i, row in enumerate(iou):
        idx = torch.where(~used)[0]
        if not len(idx):
            continue
        if torch.max(row[idx]) >= iou_threshold:
            detection[i] = True
            used[idx[torch.argmax(row[idx])]] = True
    result = torch.zeros_like(detection)
    result[order] = detection
    return result


def precision_recall(
    is_tp: torch.Tensor, scores: torch.Tensor, n_gt: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precision-recall curve of one class.

    is_tp: bool [D], scores: [D], n_gt: number of ground-truth boxes.
    Returns (precision [D], recall [D], scores sorted in decreasing order).
    """
    order = torch.argsort(scores, descending=True)
    is_tp_sorted = is_tp[order]
    is_tp_agg = torch.cumsum(is_tp_sorted, dim=0)
    k = torch.arange(1, is_tp.shape[0] + 1, device=is_tp.device)
    prec = is_tp_agg / k
    recall = is_tp_agg / n_gt
    return prec, recall, scores[order]


def average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """Area under the precision-recall curve (all-point interpolation)."""
    smoothed = torch.flip(
        torch.cummax(torch.flip(precision, dims=[0]), dim=0).values, dims=[0]
    )
    diff = torch.diff(
        recall, prepend=torch.tensor([0.0], device=recall.device)
    )
    return (smoothed * diff).sum().item()


def evaluate_detections(
    predictions: list[dict],
    ground_truth: list[dict],
    class_ids: list[int],
    iou_thresholds: list[float],
) -> tuple[pd.DataFrame, dict]:
    """Average precision per class and IoU threshold.

    predictions: one dict(boxes, scores, labels) per tile.
    ground_truth: one dict(boxes, labels) per tile, same order.
    Returns (table, curves): table with one row per class and one column per
    IoU threshold (NaN for classes without ground truth);
    curves[(class_id, iou_threshold)] = (sorted scores, precision, recall).
    """

    results = []
    curves = {}
    for class_id in class_ids:
        one_line = []
        for threshold in iou_thresholds:
            all_scores = []
            is_tp = []
            n_gt = 0
            for pred, gt in zip(predictions, ground_truth):
                boxes = pred["boxes"]
                scores = pred["scores"]
                labels = pred["labels"]

                gt_boxes = gt["boxes"]
                gt_labels = gt["labels"]
                relevant_pred = labels == class_id
                relevant_gt = gt_labels == class_id
                n_gt += relevant_gt.sum()
                rel_boxes = boxes[relevant_pred]
                rel_scores = scores[relevant_pred]

                rel_gt_boxes = gt_boxes[relevant_gt]

                matches = match_tile(
                    rel_boxes, rel_scores, rel_gt_boxes, threshold
                )
                all_scores.append(rel_scores)
                is_tp.append(matches)

            all_scores = torch.cat(all_scores)
            is_tp = torch.cat(is_tp)
            if not n_gt:
                val = float("nan")
            else:
                prec, recall, scores_sorted = precision_recall(
                    is_tp, all_scores, n_gt
                )
                val = average_precision(recall, prec)
                curves[(class_id, threshold)] = (scores_sorted, prec, recall)
            one_line.append(val)
        results.append(one_line)
    df = pd.DataFrame(results, index=class_ids, columns=iou_thresholds)
    return df, curves


def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[dict], list[dict]]:
    """Run the model over all tiles of loader.

    Returns (predictions, ground_truth): one dict per tile each, on the CPU.
    """
    model.eval()
    with torch.no_grad():
        predictions = []
        ground_truths = []
        for images, targets in tqdm(loader, desc="predicting"):
            images = [image.to(device) for image in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                predictions.append({k: v.cpu() for k, v in output.items()})
                ground_truths.append({k: v.cpu() for k, v in target.items()})
        return predictions, ground_truths


def best_threshold(
    scores: torch.Tensor,
    prec: torch.Tensor,
    recall: torch.Tensor,
    alpha: float = 0.5,
) -> tuple[float, float, float, float]:
    """Score threshold with the highest weighted F-score.

    scores, prec, recall: one entry of the curves from evaluate_detections.
    F = 1 / (alpha / P + (1 - alpha) / R); alpha = 0.5 gives F1.
    Returns (threshold, precision, recall, F) at the maximum.
    """
    f_score = 1 / (alpha / prec + (1 - alpha) / recall)
    crit = torch.argmax(f_score)
    return (
        scores[crit].item(),
        prec[crit].item(),
        recall[crit].item(),
        f_score[crit].item(),
    )
