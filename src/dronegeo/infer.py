import torch

from dronegeo.boxes import batched_nms
from dronegeo.data import get_tiles


def predict_frame(
    model: torch.nn.Module,
    image: torch.Tensor,
    size: int,
    overlap: int,
    device: torch.device,
    iou_threshold: float = 0.5,
) -> dict:
    """Detections on one full frame, merged from overlapping tiles.

    image: float32 [3, H, W] in [0, 1].
    Returns dict(boxes [N, 4] in frame pixels, scores [N], labels [N]) on
    the CPU, without a score threshold.
    """
    height, width = image.shape[1], image.shape[2]
    tiles = get_tiles(height, width, size, overlap)
    model.eval()
    all_boxes = []
    all_labels = []
    all_scores = []
    result = {}
    with torch.no_grad():
        for start_u, start_v in zip(tiles["start_u"], tiles["start_v"]):
            tile_img = image[
                :,
                start_u : start_u + size,
                start_v : start_v + size,
            ]
            output = model([tile_img.to(device)])[0]
            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]
            boxes[:, 0] = boxes[:, 0] + start_v
            boxes[:, 1] = boxes[:, 1] + start_u
            boxes[:, 2] = boxes[:, 2] + start_v
            boxes[:, 3] = boxes[:, 3] + start_u
            all_boxes.append(boxes.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())
        all_boxes = torch.cat(all_boxes)
        all_labels = torch.cat(all_labels)
        all_scores = torch.cat(all_scores)
        keep = batched_nms(all_boxes, all_scores, all_labels, iou_threshold)
        result["boxes"] = all_boxes[keep]
        result["labels"] = all_labels[keep]
        result["scores"] = all_scores[keep]
        return result
