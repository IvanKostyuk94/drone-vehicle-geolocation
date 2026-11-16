"""Detector construction (torchvision Faster R-CNN, ResNet-50 FPN v2)."""

import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batchnorm(module: nn.Module) -> None:
    """Replace every BatchNorm2d in module by a FrozenBatchNorm2d with the
    same statistics (in place)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                frozen.weight.copy_(child.weight)
                frozen.bias.copy_(child.bias)
                frozen.running_mean.copy_(child.running_mean)
                frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
        else:
            freeze_batchnorm(child)


def build_model(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """Faster R-CNN, COCO weights if pretrained, frozen batch normalisation,
    new box predictor for num_classes (4 vehicle classes + background)."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None
    )
    freeze_batchnorm(model)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    new_pred = FastRCNNPredictor(
        in_channels=in_channels, num_classes=num_classes
    )
    model.roi_heads.box_predictor = new_pred
    return model
