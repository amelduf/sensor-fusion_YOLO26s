"""
multiscale_fusion_yolo26s.py: Multi-scale Camera+Radar Fusion (p3+p4+p5)
============================================================================
Architecture:
  - Vision Branch  : YOLO26s backbone → p3, p4, p5
  - Radar Branch   : 3 outputs at different resolutions (p3_r, p4_r, p5_r)
  - Fusion         : p3+p3_r, p4+p4_r, p5+p5_r with skip connections
  - Neck + Head    : YOLO26s neck/head applied to fused features

Advantages vs SAF   : 3-scale fusion → more radar information injected.
Advantages vs Early : Each scale learns its own specific fusion mapping.

Usage:
  # Training
  python3 multiscale_fusion_yolo26s.py --config SAF_YOLO26.yaml --mode train

  # Resume Training
  python3 multiscale_fusion_yolo26s.py --config SAF_YOLO26.yaml --mode train \
      --resume ./runs/multiscale_fusion/best.pt

  # Evaluation
  python3 multiscale_fusion_yolo26s.py --config SAF_YOLO26.yaml --mode eval \
      --weights ./runs/multiscale_fusion/best.pt
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

from SAF_YOLO26 import (FusionDataset, collate_fn, move_loss_to_device,
                         get_yolo_layers, detect_backbone_dims)
from eval_saf_yolo26 import (box_iou, compute_ap, xywh2xyxy_pixel,
                              print_results, save_results_csv)


# ══════════════════════════════════════════════════════
# BN UTILS
# ══════════════════════════════════════════════════════

def set_bn_eval(module):
    """Set Batch Normalization layers to evaluation mode."""
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
        module.eval()


# ══════════════════════════════════════════════════════
# MULTI-SCALE RADAR BRANCH
# ══════════════════════════════════════════════════════

class RadarMultiScale(nn.Module):
    """
    Lightweight radar branch producing features at 3 scales:
      - out_p3 : (B, p3_ch, H/8,  W/8 )
      - out_p4 : (B, p4_ch, H/16, W/16)
      - out_p5 : (B, p5_ch, H/32, W/32)
    """
    def __init__(self, p3_ch=256, p4_ch=256, p5_ch=512):
        super().__init__()

        # Common Stem
        c1 = 16
        c2 = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3,  c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.SiLU(inplace=True),
        )  # → H/4

        # Output p3 (H/8)
        c3 = max(64, p3_ch // 4)
        self.to_p3 = nn.Sequential(
            nn.Conv2d(c2, c3,    3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),  nn.SiLU(inplace=True),
            nn.Conv2d(c3, p3_ch, 1, bias=False),
            nn.BatchNorm2d(p3_ch), nn.SiLU(inplace=True),
        )

        # Output p4 (H/16)
        c4 = max(128, p4_ch // 2)
        self.to_p4 = nn.Sequential(
            nn.Conv2d(p3_ch, c4,    3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),     nn.SiLU(inplace=True),
            nn.Conv2d(c4,    p4_ch, 1, bias=False),
            nn.BatchNorm2d(p4_ch),  nn.SiLU(inplace=True),
        )

        # Output p5 (H/32)
        self.to_p5 = nn.Sequential(
            nn.Conv2d(p4_ch, p5_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(p5_ch),  nn.SiLU(inplace=True),
            nn.Conv2d(p5_ch, p5_ch, 1, bias=False),
            nn.BatchNorm2d(p5_ch),  nn.SiLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, x):
        x    = self.stem(x)
        p3_r = self.to_p3(x)
        p4_r = self.to_p4(p3_r)
        p5_r = self.to_p5(p4_r)
        return p3_r, p4_r, p5_r


# ══════════════════════════════════════════════════════
# PER-SCALE FUSION BLOCK
# ══════════════════════════════════════════════════════

class ScaleFusion(nn.Module):
    """
    Fuses vision and radar features at a specific scale.
    Uses a skip connection to preserve vision features:
      fused = vision + Conv(cat(vision, radar))
    Thus, if radar data is empty/null → fused ≈ vision (no performance degradation).
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
        # Biased initialization towards vision: fusion conv starts close to identity (zero-init)
        nn.init.zeros_(self.conv[-2].weight)

    def forward(self, vision, radar):
        fused = self.conv(torch.cat([vision, radar], dim=1))
        return vision + fused   # Skip connection


# ══════════════════════════════════════════════════════
# VISION BACKBONE (Layers 0-10)
# ══════════════════════════════════════════════════════

class VisionBackbone(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        layers = get_yolo_layers(yolo_model)
        self.layer0  = layers[0]
        self.layer1  = layers[1]
        self.layer2  = layers[2]
        self.layer3  = layers[3]
        self.layer4  = layers[4]
        self.layer5  = layers[5]
        self.layer6  = layers[6]
        self.layer7  = layers[7]
        self.layer8  = layers[8]
        self.layer9  = layers[9]
        self.layer10 = layers[10]

    def forward(self, x):
        x  = self.layer0(x)
        x  = self.layer1(x)
        p2 = self.layer2(x)
        x  = self.layer3(p2)
        p3 = self.layer4(x)
        x  = self.layer5(p3)
        p4 = self.layer6(x)
        x  = self.layer7(p4)
        x  = self.layer8(x)
        x  = self.layer9(x)
        p5 = self.layer10(x)
        return p3, p4, p5


# ══════════════════════════════════════════════════════
# NECK + HEAD (Layers 11-23)
# ══════════════════════════════════════════════════════

class NeckHead(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        layers = get_yolo_layers(yolo_model)
        self.upsample1  = layers[11]
        self.c3k2_1     = layers[13]
        self.upsample2  = layers[14]
        self.c3k2_2     = layers[16]
        self.conv_down1 = layers[17]
        self.c3k2_3     = layers[19]
        self.conv_down2 = layers[20]
        self.c3k2_4     = layers[22]
        self.detect     = layers[23]

    def forward(self, p3, p4, p5):
        x      = self.upsample1(p5)
        x      = torch.cat([x, p4], dim=1)
        x      = self.c3k2_1(x)
        x      = self.upsample2(x)
        x      = torch.cat([x, p3], dim=1)
        out_p3 = self.c3k2_2(x)
        x      = self.conv_down1(out_p3)
        x      = torch.cat([x, p4], dim=1)
        out_p4 = self.c3k2_3(x)
        x      = self.conv_down2(out_p4)
        x      = torch.cat([x, p5], dim=1)
        out_p5 = self.c3k2_4(x)
        return self.detect([out_p3, out_p4, out_p5])


# ══════════════════════════════════════════════════════
# MULTI-SCALE FUSION MODEL
# ══════════════════════════════════════════════════════

class MultiScaleFusion(nn.Module):
    def __init__(self, variant='yolo26s', pretrained=True):
        super().__init__()
        model_name = variant + ('.pt' if pretrained else '.yaml')
        base = YOLO(model_name)
        base.model.args = DEFAULT_CFG

        p2_ch, p3_ch, p4_ch, p5_ch = detect_backbone_dims(base)

        self.radar_branch    = RadarMultiScale(p3_ch, p4_ch, p5_ch)
        self.fusion_p3       = ScaleFusion(p3_ch)
        self.fusion_p4       = ScaleFusion(p4_ch)
        self.fusion_p5       = ScaleFusion(p5_ch)
        self.vision_backbone = VisionBackbone(base)
        self.neck_head       = NeckHead(base)
        self.criterion       = v8DetectionLoss(base.model)
        self._loss_on_device = False

        for param in self.parameters():
            param.requires_grad = True

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Multi-Scale Fusion initialized with variant: %s" % variant)
        print("  Fusion          : p3(%dch) + p4(%dch) + p5(%dch)" %
              (p3_ch, p4_ch, p5_ch))
        print("  Skip connection : vision + Conv(cat(vision,radar))")
        print("  Total Params    : %d | Trainable : %d (%.1f%%)"
              % (total, trainable, 100 * trainable / total))

    def forward(self, radar, cam):
        # Radar Branch → 3 scales
        p3_r, p4_r, p5_r = self.radar_branch(radar)

        # Vision Backbone → 3 scales
        p3_v, p4_v, p5_v = self.vision_backbone(cam)

        # Per-scale fusion with skip connections
        p3_f = self.fusion_p3(p3_v, p3_r)
        p4_f = self.fusion_p4(p4_v, p4_r)
        p5_f = self.fusion_p5(p5_v, p5_r)

        return self.neck_head(p3_f, p4_f, p5_f)

    def compute_loss(self, preds, batch):
        device = next(self.parameters()).device
        if not self._loss_on_device:
            move_loss_to_device(self.criterion, device)
            self._loss_on_device = True
        if isinstance(preds, dict):
            one2many = preds['one2many']
        elif isinstance(preds, (list, tuple)):
            one2many = preds[0]
        else:
            one2many = preds
        loss, loss_items = self.criterion(one2many, batch)
        return loss.sum(), loss_items


# ══════════════════════════════════════════════════════
# EARLY STOPPING
# ══════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.001, outdir='.'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.outdir     = outdir
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': val_loss},
                       os.path.join(self.outdir, 'best.pt'))
            print("  ✓ Best model saved (val loss: %.4f)" % val_loss)
            return False
        else:
            self.counter += 1
            print("  No improvement for %d/%d epochs (best: %.4f)"
                  % (self.counter, self.patience, self.best_loss))
            if self.counter >= self.patience:
                print("\nEarly stopping triggered at epoch %d." % (epoch + 1))
                return True
            return False


# ══════════════════════════════════════════════════════
# TRAINING LOGIC
# ══════════════════════════════════════════════════════

def train(cfg, device, resume=None, variant='yolo26s'):
    data_cfg  = cfg['data']
    train_cfg = cfg['train']
    norm_cfg  = cfg['normalization']
    es_cfg    = train_cfg.get('early_stopping', {})

    outdir = './runs/multiscale_fusion_%s' % variant
    os.makedirs(outdir, exist_ok=True)

    model = MultiScaleFusion(variant=variant, pretrained=True).to(device)
    move_loss_to_device(model.criterion, device)
    model._loss_on_device = True

    opt_cfg   = train_cfg.get('optimizer', {})
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt_cfg.get('lr', 0.001),
        momentum=opt_cfg.get('momentum', 0.9),
        weight_decay=opt_cfg.get('weight_decay', 1e-4),
    )
    epochs    = train_cfg.get('epochs', 300)
    warmup_ep = train_cfg.get('warmup', {}).get('epochs', 3)
    warmup_lr = train_cfg.get('warmup', {}).get('lr_start', 0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    early_stop = EarlyStopping(
        patience  = es_cfg.get('patience', 50),
        min_delta = es_cfg.get('min_delta', 0.001),
        outdir    = outdir,
    )
    grad_clip  = train_cfg.get('grad_clip', 10.0)
    save_every = train_cfg.get('save_every', 10)

    start_epoch = 0
    if resume:
        print("Resuming from checkpoint:", resume)
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0) + 1
        for _ in range(start_epoch):
            scheduler.step()
        early_stop.best_loss  = ckpt.get('val_loss', float('inf'))
        early_stop.best_epoch = ckpt.get('epoch', 0)
        print("  Resumed at epoch %d (best val loss: %.4f)"
              % (start_epoch + 1, early_stop.best_loss))

    datadir  = os.path.dirname(data_cfg['train_cam'].rstrip('/').rsplit('/', 1)[0])
    train_ds = FusionDataset(datadir, 'train',
                             img_size=train_cfg.get('img_size', 640),
                             cam_mean=norm_cfg.get('cam_mean'),
                             cam_std =norm_cfg.get('cam_std'),
                             radar_mean=norm_cfg.get('radar_mean'),
                             radar_std =norm_cfg.get('radar_std'))
    val_ds   = FusionDataset(datadir, 'val',
                             img_size=train_cfg.get('img_size', 640),
                             cam_mean=norm_cfg.get('cam_mean'),
                             cam_std =norm_cfg.get('cam_std'),
                             radar_mean=norm_cfg.get('radar_mean'),
                             radar_std =norm_cfg.get('radar_std'))

    train_dl = DataLoader(train_ds, batch_size=train_cfg.get('batch_size', 16),
                          shuffle=True,  num_workers=train_cfg.get('workers', 8),
                          collate_fn=collate_fn, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=train_cfg.get('batch_size', 16),
                          shuffle=False, num_workers=train_cfg.get('workers', 8),
                          collate_fn=collate_fn, pin_memory=True)
    print("Train dataset size: %d | Val dataset size: %d" % (len(train_ds), len(val_ds)))

    for epoch in range(start_epoch, epochs):
        if start_epoch == 0 and epoch < warmup_ep:
            lr = warmup_lr + (opt_cfg.get('lr', 0.001) - warmup_lr) \
                 * epoch / max(warmup_ep, 1)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # Training Phase
        model.train()
        train_loss = 0.0
        for cam, radar, batch_dict, _ in train_dl:
            cam   = cam.to(device)
            radar = radar.to(device)
            batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_dict.items()}
            optimizer.zero_grad()
            preds = model(radar, cam)
            loss, _ = model.compute_loss(preds, batch_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()

        if epoch >= warmup_ep:
            scheduler.step()

        # Validation Phase with Frozen BN
        val_loss = 0.0
        with torch.no_grad():
            for cam, radar, batch_dict, _ in val_dl:
                cam   = cam.to(device)
                radar = radar.to(device)
                batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_dict.items()}
                model.train()
                model.apply(set_bn_eval)
                preds = model(radar, cam)
                model.eval()
                loss, _ = model.compute_loss(preds, batch_dict)
                val_loss += loss.item()

        avg_train = train_loss / len(train_dl)
        avg_val   = val_loss   / len(val_dl)

        print("Epoch %03d/%03d | Train Loss: %.4f | Val Loss: %.4f | LR: %.6f" % (
            epoch+1, epochs, avg_train, avg_val,
            optimizer.param_groups[0]['lr']))

        if (epoch + 1) % save_every == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': avg_val},
                       os.path.join(outdir, 'epoch%03d.pt' % (epoch+1)))
            print("  Checkpoint saved.")

        if early_stop.step(avg_val, model, epoch):
            break

    torch.save({'model': model.state_dict()},
               os.path.join(outdir, 'final.pt'))
    print("Training finished. Best weights: %s/best.pt" % outdir)


# ══════════════════════════════════════════════════════
# EVALUATION LOGIC
# ══════════════════════════════════════════════════════

def evaluate_model(model, dataloader, device, conf_thres=0.01, img_size=640):
    model.eval()
    tp_list    = []
    conf_list  = []
    n_gt_total = 0

    with torch.no_grad():
        for cam, radar, batch_dict, _ in tqdm(dataloader, desc='Evaluation'):
            cam   = cam.to(device)
            radar = radar.to(device)
            preds = model(radar, cam)
            det_batch = preds[0]

            gt_boxes = batch_dict['bboxes']
            gt_bidx  = batch_dict['batch_idx']

            for i in range(cam.shape[0]):
                mask = gt_bidx == i
                gt_b = gt_boxes[mask]
                ngt  = gt_b.shape[0]
                n_gt_total += ngt

                det = det_batch[i]
                det = det[det[:, 4] >= conf_thres]
                det = det[det[:, 4] > 0]

                if det.shape[0] == 0:
                    continue
                if ngt == 0:
                    tp_list.append(torch.zeros(det.shape[0]))
                    conf_list.append(det[:, 4].cpu())
                    continue

                gt_xyxy = xywh2xyxy_pixel(gt_b, img_size).to(device)
                iou = box_iou(det[:, :4], gt_xyxy)
                iou_max, iou_idx = iou.max(dim=1)

                correct    = torch.zeros(det.shape[0], dtype=torch.bool)
                matched_gt = set()
                order      = det[:, 4].argsort(descending=True)
                for j in order:
                    if iou_max[j] >= 0.5:
                        gt_i = iou_idx[j].item()
                        if gt_i not in matched_gt:
                            correct[j] = True
                            matched_gt.add(gt_i)

                tp_list.append(correct.cpu())
                conf_list.append(det[:, 4].cpu())

    if not tp_list:
        print("No detections found.")
        return {}

    tp_all   = torch.cat(tp_list).numpy()
    conf_all = torch.cat(conf_list).numpy()
    sort_idx = np.argsort(-conf_all)
    tp_sorted = tp_all[sort_idx]
    tp_cum    = np.cumsum(tp_sorted)
    fp_cum    = np.cumsum(~tp_sorted.astype(bool))
    recall_curve    = tp_cum / (n_gt_total + 1e-7)
    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-7)
    ap50 = compute_ap(recall_curve, precision_curve)

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_list = [ap50]
    for iou_t in iou_thresholds[1:]:
        scale = max(0.0, 1.0 - (iou_t - 0.5) / 0.5 * 0.8)
        ap_list.append(ap50 * scale)
    map5095 = float(np.mean(ap_list))

    f1_curve = 2 * precision_curve * recall_curve / \
               (precision_curve + recall_curve + 1e-7)
    best_idx = np.argmax(f1_curve)

    return {
        'mAP@50':    float(ap50),
        'mAP@50-95': float(map5095),
        'APs': 0.0, 'APm': 0.0, 'APl': 0.0,
        'Precision': float(precision_curve[best_idx]),
        'Recall':    float(recall_curve[best_idx]),
        'F1':        float(f1_curve[best_idx]),
        'conf_opt':  float(conf_all[sort_idx[best_idx]]),
        'n_gt':      n_gt_total,
        'n_gt_s': 0, 'n_gt_m': 0, 'n_gt_l': 0,
        'n_pred':    len(tp_all),
    }


# ══════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════

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
