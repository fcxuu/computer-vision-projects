"""Torchvision Faster R-CNN and pinned Ultralytics anchor-based YOLOv3."""
import hashlib
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import resize_boxes

from voc import CLASSES, unletterbox

ROOT = Path(__file__).resolve().parents[4]
YOLO_COMMIT = "34c3f2108b73031647b293a7e73603d90e5a208b"
YOLO_ASSET = "https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt"
YOLO_ASSET_SIZE = 124307969
YOLO_ASSET_SHA256 = "ba6568ba6a771d99e71546595079f93de7c69701ba483853dbefef4fc2750160"
COCO_ALIASES = {
    "aeroplane": "airplane", "motorbike": "motorcycle", "diningtable": "dining table",
    "pottedplant": "potted plant", "sofa": "couch", "tvmonitor": "tv",
}
YOLO_HYP = {
    "box": 0.05, "cls": 0.5, "obj": 1.0, "cls_pw": 1.0, "obj_pw": 1.0,
    "fl_gamma": 0.0, "label_smoothing": 0.0, "anchor_t": 4.0,
}


def category_indices(names):
    index = {label: int(i) for i, label in
             (names.items() if isinstance(names, dict) else enumerate(names))}
    return [index[COCO_ALIASES.get(label, label)] for label in CLASSES]


def faster_model(size, pretrained):
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, num_classes=91, min_size=size, max_size=size,
        rpn_pre_nms_top_n_train=320, rpn_post_nms_top_n_train=128,
        rpn_pre_nms_top_n_test=256, rpn_post_nms_top_n_test=96,
        box_batch_size_per_image=64, box_score_thresh=0.005, box_detections_per_img=100,
    )
    if pretrained:
        model.load_state_dict(weights.get_state_dict(check_hash=True), strict=True)
    old = model.roi_heads.box_predictor
    head = FastRCNNPredictor(old.cls_score.in_features, len(CLASSES) + 1)
    if pretrained:
        indices = torch.as_tensor([0, *category_indices(weights.meta["categories"])])
        with torch.no_grad():
            head.cls_score.weight.copy_(old.cls_score.weight[indices])
            head.cls_score.bias.copy_(old.cls_score.bias[indices])
            head.bbox_pred.weight.copy_(old.bbox_pred.weight.reshape(91, 4, -1)[indices].flatten(0, 1))
            head.bbox_pred.bias.copy_(old.bbox_pred.bias.reshape(91, 4)[indices].flatten())
    model.roi_heads.box_predictor = head
    frozen = [block for name, block in model.backbone.body.items() if int(name) < 13]
    return model, frozen


def load_yolo_modules(vendor):
    path = vendor.resolve()
    actual_commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != YOLO_COMMIT:
        raise ValueError(f"Use the pinned YOLOv3 commit {YOLO_COMMIT}; checkout contains {actual_commit}")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / "voc2007" / "ultralytics-config"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    sys.path.insert(0, str(path))
    from models.yolo import Model
    from utils.general import non_max_suppression
    from utils.loss import ComputeLoss

    return Model, non_max_suppression, ComputeLoss


def read_trusted_yolo_weights(path):
    if path.stat().st_size != YOLO_ASSET_SIZE:
        raise ValueError(f"Unexpected size for {path}. Download the pinned model from {YOLO_ASSET}")
    with path.open("rb") as source:
        actual_sha = hashlib.file_digest(source, "sha256").hexdigest()
    if actual_sha != YOLO_ASSET_SHA256:
        raise ValueError(f"Unexpected SHA-256 for the pinned official YOLOv3 weights: {actual_sha}")
    # The pinned vendor archive serializes a legacy Ultralytics Model object.
    data = torch.load(path, map_location="cpu", weights_only=False)
    model = data.get("ema") or data["model"]
    return model.float()


def yolo_model(vendor, pretrained_path):
    Model, nms, loss = load_yolo_modules(vendor)
    model = Model(str(vendor / "models" / "yolov3.yaml"), nc=len(CLASSES))
    model.names = list(CLASSES)
    model.hyp = YOLO_HYP.copy()
    if pretrained_path is not None:
        source = read_trusted_yolo_weights(pretrained_path)
        classes = category_indices(source.names)
        before, after = source.model[-1], model.model[-1]
        state = source.state_dict()
        initial = model.state_dict()
        shared = {key: weights for key, weights in state.items()
                  if key in initial and initial[key].shape == weights.shape}
        missing, _ = model.load_state_dict(shared, strict=False)
        allowed = {f"model.{len(model.model) - 1}.m.{index}.{kind}"
                   for index in range(len(after.m)) for kind in ("weight", "bias")}
        if set(missing) != allowed or before.na != after.na:
            raise ValueError("Official YOLOv3 pretrained checkpoint does not match the pinned architecture")
        output_indices = [*range(5), *(5 + index for index in classes)]
        with torch.no_grad():
            for source_head, target_head in zip(before.m, after.m):
                channels = source_head.weight.reshape(before.na, before.no, *source_head.weight.shape[1:])
                target_head.weight.copy_(channels[:, output_indices].flatten(0, 1))
                bias = source_head.bias.reshape(before.na, before.no)
                target_head.bias.copy_(bias[:, output_indices].flatten())
    return model, list(model.model[:7]), nms, loss


class HostROIAlign(nn.Module):
    """Run ROIAlign on CPU while preserving gradients to MPS feature maps."""

    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def forward(self, features, boxes, image_shapes):
        device = next(iter(features.values())).device
        host = {name: feature.cpu() for name, feature in features.items()}
        return self.pool(host, [box.cpu() for box in boxes], image_shapes).to(device)


class Detector:
    def __init__(self, name, size, device, vendor, data, pretrained=False):
        self.name, self.size, self.device = name, size, device
        if name == "faster_rcnn":
            torch.hub.set_dir(str(data / "torch-hub"))
            self.net, self.frozen = faster_model(size, pretrained)
            if device.type == "mps":
                self.net.roi_heads.box_roi_pool = HostROIAlign(self.net.roi_heads.box_roi_pool)
            self.net.to(device)
        elif name == "yolov3":
            source = data / "pretrained" / "yolov3.pt" if pretrained else None
            net, self.frozen, self.nms, loss = yolo_model(vendor, source)
            self.net = net.to(device)
            self.yolo_loss = loss(self.net)
        else:
            raise ValueError(f"Unknown detector {name}")
        for module in self.frozen:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def training_mode(self):
        self.net.train()
        for module in self.frozen:
            module.eval()
        # Small detection batches share the pretrained BN statistics.
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def targets(self, records):
        if self.name == "faster_rcnn":
            return [
                {"boxes": r["boxes"][~r["difficult"]].to(self.device),
                 "labels": (r["labels"][~r["difficult"]] + 1).to(self.device)}
                for r in records
            ]
        rows = []
        for index, record in enumerate(records):
            keep = ~record["difficult"]
            xyxy = record["boxes"][keep] / self.size
            xywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), dim=1)
            labels = record["labels"][keep].float()[:, None]
            rows.append(torch.cat((torch.full_like(labels, index), labels, xywh), dim=1))
        return torch.cat(rows).to(self.device)

    def training_loss(self, images, targets):
        if self.name == "faster_rcnn":
            return sum(self.net(list(images), targets).values())
        loss, _ = self.yolo_loss(self.net(images), targets)
        return loss.squeeze() / len(images)

    def decode_yolo(self, raw):
        predictions = self.nms(raw, conf_thres=0.005, iou_thres=0.5, multi_label=True, max_det=100)
        return [{"boxes": item[:, :4], "scores": item[:, 4], "labels": item[:, 5].long()}
                for item in predictions]

    def outputs(self, images, targets=None):
        self.net.eval()
        if self.name == "yolov3":
            raw, maps = self.net(images)
            loss = None
            if targets is not None:
                value, _ = self.yolo_loss(maps, targets)
                loss = value.squeeze() / len(images)
            return self.decode_yolo(raw), loss
        prediction = self.net(list(images))
        loss = None
        if targets is not None:
            self.training_mode()
            loss = self.training_loss(images, targets)
        for item in prediction:
            item["labels"] -= 1
        self.net.eval()
        return prediction, loss

    def proposal_boxes(self, image, limit=24):
        if self.name != "faster_rcnn":
            raise ValueError("Region proposals are available only for Faster R-CNN")
        self.net.eval()
        transformed, _ = self.net.transform([image], None)
        features = self.net.backbone(transformed.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict({"0": features})
        proposals, _ = self.net.rpn(transformed, features, None)
        return resize_boxes(proposals[0][:limit], transformed.image_sizes[0], image.shape[-2:])


def original_detections(outputs, records):
    predictions = {}
    for output, record in zip(outputs, records):
        raw = unletterbox(output["boxes"].detach().cpu().numpy(), record["geometry"])
        size = raw[:, 2:] - raw[:, :2]
        keep = (size > 0).all(axis=1)
        predictions[record["id"]] = {
            "boxes": raw[keep], "labels": output["labels"].detach().cpu().numpy()[keep],
            "scores": output["scores"].detach().cpu().numpy().astype(np.float32)[keep],
        }
    return predictions
