import torch


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Intersection over union of every pair of boxes.

    boxes_a: [N, 4], boxes_b: [M, 4], both (x1, y1, x2, y2).
    Returns [N, M], entry (i, j) = IoU(a_i, b_j).
    """
    a_area = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    b_area = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ar_sum = a_area[None, :] + b_area[:, None]

    left = torch.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    up = torch.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    right = torch.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    down = torch.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    intersect = (right - left).clamp(min=0) * (down - up).clamp(min=0)
    union = ar_sum.T - intersect
    iou = torch.div(intersect, union)
    return iou


def nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Non-maximum suppression.

    boxes: [N, 4], scores: [N].
    Returns int64 indices of the kept boxes, sorted by decreasing score.
    """
    order = torch.argsort(scores, descending=True)
    boxes_sorted = boxes[order]
    iou = box_iou(boxes_sorted, boxes_sorted)
    removed = torch.zeros(
        scores.shape[0], dtype=torch.bool, device=boxes.device
    )
    overlap = iou > iou_threshold
    keep = []
    for i in range(scores.shape[0]):
        if removed[i]:
            continue
        keep.append(i)
        removed = torch.logical_or(removed, overlap[i, :])
    return order[keep]


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Class-aware non-maximum suppression.

    boxes: [N, 4], scores: [N], labels: [N].
    Returns int64 indices of the kept boxes, sorted by decreasing score.
    """
    indices = torch.arange(labels.shape[0], device=labels.device)
    all_keep = []
    for label in labels.unique():
        same_label = labels == label
        keep = nms(boxes[same_label, :], scores[same_label], iou_threshold)
        all_keep.extend(indices[same_label][keep].tolist())
    order = torch.argsort(scores[all_keep], descending=True)
    return torch.tensor(all_keep, dtype=torch.int64, device=boxes.device)[
        order
    ]
