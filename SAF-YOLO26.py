"""
SAF-YOLO26: Spatial Attention Fusion + YOLO26
===============================================
Hybrid Sensor Fusion Architecture for Object Detection.
Compatible with yolo26n, yolo26s, yolo26m (automatic dimension detection).

Architecture:
  - Radar Branch : Lightweight feature extractor (2 Conv + 1 Bottleneck).
  - SAF Block    : Multi-kernel (1x1, 3x3, 5x5) Spatial Attention generation.
  - Vision Branch: Full pre-trained YOLO26 backbone.
  - Fusion       : Element-wise multiplication (vision_feat ⊗ attention).
  - Neck + Head  : Standard YOLO26 neck and detection head.

Reference: "Spatial Attention Fusion for Obstacle Detection Using
             MmWave Radar and Vision Sensor", Chang et al., Sensors 2020
"""

import os
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import DEFAULT_CFG


# ══════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════

def get_yolo_layers(yolo_model):
    """Extracts the underlying list of layers from an Ultralytics YOLO model."""
    sequential = list(yolo_model.model.children())[0]
    return list(sequential.children())


def detect_backbone_dims(yolo_model):
    """
    Passes a dummy tensor through the backbone to automatically 
    detect the number of channels at P2, P3, P4, and P5 stages.
    """
    layers = get_yolo_layers(yolo_model)
    x = torch.randn(1, 3, 640, 640)
    yolo_model.model.eval()
    with torch.no_grad():
        curr = x
        dims = {}
        for i, layer in enumerate(layers[:11]):
            curr = layer(curr)
            if i in [2, 4, 6, 10]:
                dims[i] = curr.shape[1]
    
    p2_ch, p3_ch, p4_ch, p5_ch = dims[2], dims[4], dims[6], dims[10]
    print(f"  Detected backbone dims: p2={p2_ch} p3={p3_ch} p4={p4_ch} p5={p5_ch}")
    return p2_ch, p3_ch, p4_ch, p5_ch


def move_loss_to_device(criterion, device):
    """
    Ensures all internal tensors (gain, anchors, etc.) of the v8DetectionLoss 
    are moved to the correct device (CPU/CUDA).
    """
    criterion.device = device

    def _move_obj(obj, visited=None):
        if visited is None:
            visited = set()
        obj_id = id(obj)
        if obj_id in visited or obj is None or not hasattr(obj, '__dict__'):
            return
        visited.add(obj_id)
        for name, val in vars(obj).items():
            if isinstance(val, torch.Tensor):
                setattr(obj, name, val.to(device))
            elif hasattr(val, '__dict__'):
                _move_obj(val, visited)

    _move_obj(criterion)


def set_bn_eval(module):
    """
    Forces BatchNorm layers into evaluation mode.
    Used during validation to stabilize loss while keeping the Detect head 
    in 'train' mode to output the structure required by the loss function.
    """
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
        module.eval()


# ══════════════════════════════════════════════════════
# 1. LIGHTWEIGHT RADAR BRANCH
# ══════════════════════════════════════════════════════

class RadarBottleneck(nn.Module):
    """Standard residual bottleneck for the radar feature extractor."""
    def __init__(self, channels):
        super().__init__()
        self.cv1 = nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.cv2 = nn.Conv2d(channels // 2, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return x + self.act(self.bn2(self.cv2(self.act(self.bn1(self.cv1(x))))))


class RadarBranch(nn.Module):
    """
    Downsamples the radar projection image to match the spatial dimensions 
    of the vision backbone's P5 feature map.
    """
    def __init__(self, out_ch=256):
        super().__init__()
        c1 = max(16, out_ch // 16)
        c2 = max(32, out_ch // 8)
        c3 = max(64, out_ch // 4)
        c4 = max(128, out_ch // 2)

        self.stem = nn.Sequential(
            nn.Conv2d(3,  c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.block = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
            RadarBottleneck(c3),
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c4, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.proj(self.block(self.stem(x)))


# ══════════════════════════════════════════════════════
# 2. SPATIAL ATTENTION FUSION (SAF) BLOCK
# ══════════════════════════════════════════════════════

class SAFBlock(nn.Module):
    """
    Generates a spatial attention map by applying convolutions with 
    different receptive fields to the radar features.
    """
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, radar_feat):
        # Merge multi-scale information and squash to [0, 1] range
        return self.sigmoid(
            self.conv1x1(radar_feat) +
            self.conv3x3(radar_feat) +
            self.conv5x5(radar_feat)
        )


# ══════════════════════════════════════════════════════
# 3. VISION BACKBONE (Layers 0-10)
# ══════════════════════════════════════════════════════

class VisionBackbone(nn.Module):
    """Wraps the YOLO backbone layers to extract intermediate feature maps (P2-P5)."""
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
        return p2, p3, p4, p5


# ══════════════════════════════════════════════════════
# 4. NECK + HEAD (Layers 11-23)
# ══════════════════════════════════════════════════════

class NeckHead(nn.Module):
    """Wraps the YOLO neck (PANet) and detection head."""
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

    def forward(self, fused, p2, p3, p4):
        x      = self.upsample1(fused)
        x      = torch.cat([x, p4], dim=1)
        x      = self.c3k2_1(x)
        x      = self.upsample2(x)
        x      = torch.cat([x, p3], dim=1)
        out_p3 = self.c3k2_2(x)
        x      = self.conv_down1(out_p3)
        x      = torch.cat([x, p4], dim=1)
        out_p4 = self.c3k2_3(x)
        x      = self.conv_down2(out_p4)
        x      = torch.cat([x, fused], dim=1)
        out_p5 = self.c3k2_4(x)
        return self.detect([out_p3, out_p4, out_p5])


# ══════════════════════════════════════════════════════
# 5. COMPLETE SAF-YOLO26 MODEL
# ══════════════════════════════════════════════════════

class SAFYolo26(nn.Module):
    def __init__(self, variant='yolo26s', num_classes=1, pretrained=True):
        super().__init__()
        model_name = variant + ('.pt' if pretrained else '.yaml')
        base = YOLO(model_name)
        base.model.args = DEFAULT_CFG # Required for loss initialization

        p2_ch, p3_ch, p4_ch, p5_ch = detect_backbone_dims(base)

        self.radar_branch    = RadarBranch(out_ch=p5_ch)
        self.saf_block       = SAFBlock(in_channels=p5_ch)
        self.vision_backbone = VisionBackbone(base)
        self.neck_head       = NeckHead(base)
        self.criterion       = v8DetectionLoss(base.model)
        self._loss_on_device = False

        print(f"SAF-YOLO26 initialized using {variant}.")
        print(f"  Radar Branch: Outputting {p5_ch} channels")

    def forward(self, radar_img, vision_img):
        # Extract radar features and attention mask
        radar_feat = self.radar_branch(radar_img)
        attention  = self.saf_block(radar_feat)
        
        # Extract vision features
        p2, p3, p4, p5 = self.vision_backbone(vision_img)
        
        # Squeeze vision features using radar-guided attention
        fused = p5 * attention
        
        return self.neck_head(fused, p2, p3, p4)

    def compute_loss(self, preds, batch):
        """Processes predictions and computes the YOLO multi-task loss."""
        device = next(self.parameters()).device
        if not self._loss_on_device:
            move_loss_to_device(self.criterion, device)
            self._loss_on_device = True
        
        one2many = preds[0] if isinstance(preds, (list, tuple)) else preds
        loss, loss_items = self.criterion(one2many, batch)
        return loss.sum(), loss_items


# ══════════════════════════════════════════════════════
# 6. MULTIMODAL FUSION DATASET
# ══════════════════════════════════════════════════════

class FusionDataset(Dataset):
    """
    Dataset loader for synchronized Camera (RGB) and Radar (encoded PNG) images.
    Expects YOLO-formatted labels (.txt).
    """
    def __init__(self, root, split='train', img_size=640,
                 cam_mean=None, cam_std=None,
                 radar_mean=None, radar_std=None):
        self.img_dir   = os.path.join(root, 'images',       split)
        self.radar_dir = os.path.join(root, 'images_radar', split)
        self.label_dir = os.path.join(root, 'labels',       split)
        self.img_size  = img_size

        self.samples = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])

        # Default Normalization (ImageNet for camera)
        self.transform_cam = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=cam_mean or [0.485, 0.456, 0.406], 
                        std=cam_std or [0.229, 0.224, 0.225]),
        ])
        self.transform_radar = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=radar_mean or [0.0, 0.0, 0.0], 
                        std=radar_std or [1.0, 1.0, 1.0]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        base  = os.path.splitext(fname)[0]

        # Load RGB
        cam   = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        cam   = self.transform_cam(cam)

        # Load encoded Radar image
        radar = Image.open(os.path.join(self.radar_dir, base + '.png')).convert('RGB')
        radar = self.transform_radar(radar)

        # Load YOLO labels
        label_path = os.path.join(self.label_dir, base + '.txt')
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    vals = [float(x) for x in line.strip().split()]
                    if len(vals) == 5: labels.append(vals)
        
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        return cam, radar, labels, fname


def collate_fn(batch):
    """Custom collate to build the batch dictionary required by Ultralytics Loss."""
    cams, radars, labels, fnames = zip(*batch)
    cams, radars = torch.stack(cams), torch.stack(radars)

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
        'img': cams,
    }
    return cams, radars, batch_dict, list(fnames)


# ══════════════════════════════════════════════════════
# 7. EARLY STOPPING & TRAINING
# ══════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, outdir='.'):
        self.patience, self.min_delta = patience, min_delta
        self.outdir, self.best_loss = outdir, float('inf')
        self.counter, self.best_epoch = 0, 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.counter, self.best_epoch = val_loss, 0, epoch
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_loss': val_loss}, 
                       os.path.join(self.outdir, 'best.pt'))
            print(f"  ✓ Model improved. Saved to {self.outdir}/best.pt")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                return True
            return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='saf_yolo26_config.yaml')
    parser.add_argument('--variant', default='yolo26s', choices=['yolo26n', 'yolo26s', 'yolo26m'])
    parser.add_argument('--resume',  default=None)
    args = parser.parse_args()

    # Load YAML Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outdir = cfg['train'].get('outdir', './runs/saf_yolo26')
    os.makedirs(outdir, exist_ok=True)

    # Initialize Model
    model = SAFYolo26(variant=args.variant, num_classes=cfg['data']['nc']).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])

    # DataLoaders
    datadir = os.path.dirname(cfg['data']['train_cam'].rstrip('/').rsplit('/', 1)[0])
    train_dl = DataLoader(FusionDataset(datadir, 'train'), batch_size=cfg['train']['batch_size'], 
                          shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dl = DataLoader(FusionDataset(datadir, 'val'), batch_size=cfg['train']['batch_size'], 
                        shuffle=False, collate_fn=collate_fn, num_workers=4)

    early_stop = EarlyStopping(patience=15, outdir=outdir)

    # Training Loop
    for epoch in range(cfg['train']['epochs']):
        # Train Phase
        model.train()
        for cam, rad, batch_dict, _ in train_dl:
            cam, rad = cam.to(device), rad.to(device)
            batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
            
            optimizer.zero_grad()
            preds = model(rad, cam)
            loss, _ = model.compute_loss(preds, batch_dict)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation Phase
        model.train() # Keep in train to allow loss calculation
        model.apply(set_bn_eval) # But freeze BN stats
        val_loss = 0
        with torch.no_grad():
            for cam, rad, batch_dict, _ in val_dl:
                cam, rad = cam.to(device), rad.to(device)
                batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                preds = model(rad, cam)
                loss, _ = model.compute_loss(preds, batch_dict)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_dl)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val:.4f}")
        
        if early_stop.step(avg_val, model, epoch):
            break

    print("Training finished.")
