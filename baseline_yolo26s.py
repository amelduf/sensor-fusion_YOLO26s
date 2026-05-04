"""
baseline_yolo26s.py : Standard YOLO26s Training and Evaluation (Vision-only)
==================================================================================
Provides a performance baseline to compare against SAF-YOLO26s (Camera+Radar fusion).

Usage:
  # Training
  python3 baseline_yolo26s.py --config SAF_YOLO26.yaml --mode train

  # Evaluation
  python3 baseline_yolo26s.py --config SAF_YOLO26.yaml --mode eval \
      --weights ./runs/baseline_yolo26s/best.pt

  # Comparison with SAF Model
  python3 baseline_yolo26s.py --config SAF_YOLO26.yaml --mode compare \
      --weights ./runs/baseline_yolo26s/best.pt \
      --saf_weights ./runs/saf_yolo26/best.pt
"""

import os
import csv
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

# Importing shared utilities from the SAF-YOLO26 project
from eval_saf_yolo26 import (box_iou, compute_ap, xywh2xyxy_pixel,
                              print_results, save_results_csv)
from SAF_YOLO26 import move_loss_to_device


# ══════════════════════════════════════════════════════
# BATCH NORM UTILITY
# ══════════════════════════════════════════════════════

def set_bn_eval(module):
    """Sets all BatchNorm layers to evaluation mode (using running stats)."""
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
        module.eval()


# ══════════════════════════════════════════════════════
# VISION-ONLY DATASET
# ══════════════════════════════════════════════════════

class CamDataset(Dataset):
    """Standard RGB-only dataset for baseline evaluation."""
    def __init__(self, root, split='train', img_size=640,
                 cam_mean=None, cam_std=None):
        self.img_dir   = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'labels', split)
        self.img_size  = img_size

        self.samples = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])

        # ImageNet normalization as default
        cam_mean = cam_mean or [0.485, 0.456, 0.406]
        cam_std  = cam_std  or [0.229, 0.224, 0.225]

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=cam_mean, std=cam_std),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        base  = os.path.splitext(fname)[0]

        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        img = self.transform(img)

        # Load YOLO-format label
        label_path = os.path.join(self.label_dir, base + '.txt')
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    vals = [float(x) for x in line.strip().split()]
                    if len(vals) == 5:
                        labels.append(vals)
        
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        return img, labels, fname


def collate_fn_cam(batch):
    """Batches images and aligns class labels for detection loss."""
    imgs, labels, fnames = zip(*batch)
    imgs = torch.stack(imgs)

    cls_list, bbox_list, bidx_list = [], [], []
    for i, lab in enumerate(labels):
        if lab.shape[0] > 0:
            cls_list.append(lab[:, 0])
            bbox_list.append(lab[:, 1:])
            bidx_list.append(torch.full((lab.shape[0],), float(i)))

    batch_dict = {
        'cls': torch.cat(cls_list) if cls_list else torch.zeros(0),
        'bboxes': torch.cat(bbox_list) if bbox_list else torch.zeros((0, 4)),
        'batch_idx': torch.cat(bidx_list) if bidx_list else torch.zeros(0),
        'img': imgs,
    }
    return imgs, batch_dict, list(fnames)


# ══════════════════════════════════════════════════════
# BASELINE YOLO26s MODEL
# ══════════════════════════════════════════════════════

class BaselineYolo26s(torch.nn.Module):
    """Standard YOLOv8-style architecture wrapper for camera-only training."""
    def __init__(self, pretrained=True):
        super().__init__()
        base = YOLO('yolo26s.pt' if pretrained else 'yolo26s.yaml')
        base.model.args = DEFAULT_CFG
        self.model     = base.model
        self.criterion = v8DetectionLoss(base.model)
        self._loss_on_device = False

        print("Baseline YOLO26s Initialized.")
        print(f"  Total Params: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, preds, batch):
        device = next(self.parameters()).device
        if not self._loss_on_device:
            move_loss_to_device(self.criterion, device)
            self._loss_on_device = True
        
        # Extract correct tensor for loss calculation based on model output type
        one2many = preds['one2many'] if isinstance(preds, dict) else (preds[0] if isinstance(preds, (list, tuple)) else preds)
        loss, loss_items = self.criterion(one2many, batch)
        return loss.sum(), loss_items


# ══════════════════════════════════════════════════════
# TRAINING & EVALUATION LOGIC
# ══════════════════════════════════════════════════════

def train_baseline(cfg, device, resume=None):
    """Main training loop for the vision-only baseline model."""
    data_cfg, train_cfg, norm_cfg = cfg['data'], cfg['train'], cfg['normalization']
    outdir = './runs/baseline_yolo26s'
    os.makedirs(outdir, exist_ok=True)

    model = BaselineYolo26s(pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.get('epochs', 300))

    # Setup Dataset/DataLoader
    datadir = os.path.dirname(data_cfg['train_cam'].rstrip('/').rsplit('/', 1)[0])
    train_ds = CamDataset(datadir, 'train', img_size=640)
    val_ds   = CamDataset(datadir, 'val', img_size=640)
    train_dl = DataLoader(train_ds, batch_size=train_cfg.get('batch_size', 16), shuffle=True, collate_fn=collate_fn_cam)
    val_dl   = DataLoader(val_ds, batch_size=train_cfg.get('batch_size', 16), shuffle=False, collate_fn=collate_fn_cam)

    for epoch in range(train_cfg.get('epochs', 300)):
        # Training Phase
        model.train()
        train_loss = 0.0
        for imgs, batch_dict, _ in train_dl:
            imgs = imgs.to(device)
            batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss, _ = model.compute_loss(preds, batch_dict)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation Phase
        model.train()
        model.apply(set_bn_eval) # Keep model in train mode for loss but freeze BN stats
        val_loss = 0.0
        with torch.no_grad():
            for imgs, batch_dict, _ in val_dl:
                imgs = imgs.to(device)
                batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                preds = model(imgs)
                loss, _ = model.compute_loss(preds, batch_dict)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_dl):.4f} | Val Loss: {val_loss/len(val_dl):.4f}")


def evaluate_baseline(model, dataloader, device, conf_thres=0.01, img_size=640):
    """Computes mAP metrics for the baseline model."""
    model.eval()
    tp_list, conf_list, n_gt_total = [], [], 0

    with torch.no_grad():
        for imgs, batch_dict, _ in tqdm(dataloader, desc='Baseline Evaluation'):
            imgs = imgs.to(device)
            preds = model(imgs)
            det_batch = preds[0] # Typical Ultralytics output structure (B, 300, 6)

            gt_boxes, gt_bidx = batch_dict['bboxes'], batch_dict['batch_idx']

            for i in range(imgs.shape[0]):
                mask = gt_bidx == i
                gt_b = gt_boxes[mask]
                n_gt_total += gt_b.shape[0]

                det = det_batch[i]
                det = det[det[:, 4] >= conf_thres]

                if det.shape[0] == 0: continue

                gt_xyxy = xywh2xyxy_pixel(gt_b, img_size).to(device)
                iou = box_iou(det[:, :4], gt_xyxy)
                iou_max, iou_idx = iou.max(dim=1)

                correct, matched_gt = torch.zeros(det.shape[0], dtype=torch.bool), set()
                order = det[:, 4].argsort(descending=True)
                
                for j in order:
                    if iou_max[j] >= 0.5:
                        gt_i = iou_idx[j].item()
                        if gt_i not in matched_gt:
                            correct[j] = True
                            matched_gt.add(gt_i)

                tp_list.append(correct.cpu())
                conf_list.append(det[:, 4].cpu())

    # Metric calculations (Precision, Recall, F1, mAP)
    tp_all = torch.cat(tp_list).numpy()
    conf_all = torch.cat(conf_list).numpy()
    sort_idx = np.argsort(-conf_all)
    tp_sorted = tp_all[sort_idx]
    
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(~tp_sorted.astype(bool))
    recall_curve = tp_cum / (n_gt_total + 1e-7)
    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-7)
    ap50 = compute_ap(recall_curve, precision_curve)

    return {
        'mAP@50': float(ap50),
        'Precision': float(precision_curve.max()),
        'Recall': float(recall_curve.max()),
        'n_gt': n_gt_total,
    }


# ══════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════

def print_comparison(baseline_res, saf_res):
    """Prints a comparison table between vision-only and sensor fusion performance."""
    print("\n" + "═" * 75)
    print("  COMPARISON: YOLO26s (Vision-Only) vs SAF-YOLO26s (Fusion)")
    print("═" * 75)
    print("  %-20s │ %-18s │ %-18s │ %-10s" % ('Metric', 'Baseline', 'SAF-Fusion', 'Delta'))
    print("─" * 75)

    metrics = ['mAP@50', 'Precision', 'Recall']
    for m in metrics:
        b, s = baseline_res[m], saf_res[m]
        delta = s - b
        print("  %-20s │ %-18.4f │ %-18.4f │ %+.4f" % (m, b, s, delta))
    print("═" * 75)


if __name__ == '__main__':
    import argparse
    # Argparse and logic for Train/Eval/Compare modes (similar to the logic in main code)
    # [...]
