"""
early_fusion_yolo26s.py : Early Fusion YOLO26s (Camera + Radar Input)
=========================================================================
Simple Architecture: Concatenate cam + radar -> Conv 6→3ch -> Full YOLO26s.
The backbone learns to interpret the radar signal itself.
"""

import os
import csv
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

# Internal Project Imports
from SAF_YOLO26 import (FusionDataset, collate_fn, move_loss_to_device,
                         get_yolo_layers)
from eval_saf_yolo26 import (box_iou, compute_ap, xywh2xyxy_pixel,
                              print_results, save_results_csv)

# ══════════════════════════════════════════════════════
# EARLY FUSION MODEL
# ══════════════════════════════════════════════════════

class EarlyFusionYolo26s(nn.Module):
    """
    Early Fusion: Concatenated cam + radar (6ch) → Conv 6→32ch → YOLO26s.

    The input Conv is initialized to give more weight to the camera 
    (YOLO26s pre-trained weights) and less to the radar (small random weights).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        base = YOLO('yolo26s.pt' if pretrained else 'yolo26s.yaml')
        base.model.args = DEFAULT_CFG

        # ── Modify Input Conv (6 channels) ──
        # Replacing the first YOLO layer (3→32ch) with a (6→32ch) layer
        layers = get_yolo_layers(base)
        first_conv_block = layers[0]   # This is the 'Stem' Conv block

        old_conv = first_conv_block.conv   # Original nn.Conv2d(3, 32, ...)
        
        # Define new Conv with 6 input channels
        new_conv = nn.Conv2d(
            in_channels=6, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride,
            padding=old_conv.padding, 
            bias=False
        )

        # ── Weight Initialization ──
        with torch.no_grad():
            # Copy pre-trained camera weights into the first 3 input channels
            new_conv.weight[:, :3, :, :] = old_conv.weight.data.clone()
            # Initialize radar channels with small random values
            nn.init.normal_(new_conv.weight[:, 3:, :, :], mean=0.0, std=0.01)

        # Swap the layer inside the YOLO model
        first_conv_block.conv = new_conv

        self.model = base.model
        self.criterion = v8DetectionLoss(base.model)
        self._loss_on_device = False

        print("Early Fusion YOLO26s Initialized.")
        print(f"  Total Params: {sum(p.numel() for p in self.parameters())}")

    def forward(self, radar, cam):
        # Concatenate camera + radar on the channel dimension (Dim 1)
        x = torch.cat([cam, radar], dim=1)   # Shape: (Batch, 6, Height, Width)
        return self.model(x)

    def compute_loss(self, preds, batch):
        device = next(self.parameters()).device
        if not self._loss_on_device:
            move_loss_to_device(self.criterion, device)
            self._loss_on_device = True
        
        # Extract correct output structure for loss calculation
        one2many = preds['one2many'] if isinstance(preds, dict) else (preds[0] if isinstance(preds, (list, tuple)) else preds)
        loss, loss_items = self.criterion(one2many, batch)
        return loss.sum(), loss_items

# ══════════════════════════════════════════════════════
# EVALUATION & COMPARISON
# ══════════════════════════════════════════════════════

def print_comparison_3(baseline_res, saf_res, early_res):
    """Prints a comparison table for the three fusion strategies."""
    print("\n" + "═" * 85)
    print("  FUSION STRATEGY PERFORMANCE COMPARISON")
    print("═" * 85)
    print("  %-20s │ %-18s │ %-18s │ %-18s" %
          ('Metric', 'YOLO26s (Cam Only)', 'SAF-Fusion (Mid)', 'Early Fusion (Input)'))
    print("─" * 85)
    for m in ['mAP@50', 'Precision', 'Recall', 'F1']:
        b, s, e = baseline_res[m], saf_res[m], early_res[m]
        best = max(b, s, e)
        def fmt(v):
            marker = ' [BEST]' if v == best else '       '
            return f"{v:.4f} ({v*100:.1f}%){marker}"
        print("  %-20s │ %-20s │ %-20s │ %-20s" % (m, fmt(b), fmt(s), fmt(e)))
    print("═" * 85)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='SAF_YOLO26.yaml')
    parser.add_argument('--variant', default='yolo26s',
                        choices=['yolo26n', 'yolo26s', 'yolo26m'])
    parser.add_argument('--mode',    default='train',
                        choices=['train', 'eval'])
    parser.add_argument('--weights', default='./runs/multiscale_fusion/best.pt')
    parser.add_argument('--resume',  default=None)
    parser.add_argument('--split',   default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--conf',    type=float, default=0.01)
    parser.add_argument('--batch',   type=int,   default=16)
    parser.add_argument('--workers', type=int,   default=4)
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device :", device)
    print("Mode   :", args.mode)

    if args.mode == 'train':
        train(cfg, device, resume=args.resume, variant=args.variant)

    elif args.mode == 'eval':
        data_cfg = cfg['data']
        norm_cfg = cfg['normalization']
        model = MultiScaleFusion(variant=args.variant, pretrained=False).to(device)
        ckpt  = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['model'])
        print("Weights loaded from epoch %d" % (ckpt.get('epoch', 0) + 1))

        datadir = os.path.dirname(
            data_cfg['train_cam'].rstrip('/').rsplit('/', 1)[0])
        dataset = FusionDataset(datadir, args.split,
                                img_size=cfg['train'].get('img_size', 640),
                                cam_mean=norm_cfg.get('cam_mean'),
                                cam_std =norm_cfg.get('cam_std'),
                                radar_mean=norm_cfg.get('radar_mean'),
                                radar_std =norm_cfg.get('radar_std'))
        dl = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn,
                        pin_memory=True)
        print("Evaluating on %d images (%s split)" % (len(dataset), args.split))

        results = evaluate_model(model, dl, device,
                                 conf_thres=args.conf,
                                 img_size=cfg['train'].get('img_size', 640))
        if results:
            print_results(results, variant='multiscale_fusion', split=args.split)
            os.makedirs('./runs/eval', exist_ok=True)
            save_results_csv(results,
                             './runs/eval/results_multiscale_%s.csv' % args.split,
                             'multiscale_fusion', args.split)
