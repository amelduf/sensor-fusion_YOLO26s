"""
label_utils.py: YOLO label parser supporting both Detection AND Segmentation
=============================================================================
Automatically converts polygon segmentation labels into bounding boxes.

Detection Format    : cls cx cy w h          (5 values)
Segmentation Format : cls x1 y1 x2 y2 ...  (N coordinate pairs)
"""

import os
import torch


def read_labels(label_path):
    """
    Reads a YOLO label file and returns a (N, 5) tensor: cls cx cy w h.
    Supports:
      - Detection format    : cls cx cy w h
      - Segmentation format : cls x1 y1 x2 y2 ... → converted to bbox
    """
    labels = []

    if not os.path.isfile(label_path):
        return torch.zeros((0, 5), dtype=torch.float32)

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals = [float(x) for x in line.split()]
            except ValueError:
                continue

            if len(vals) == 5:
                # Standard YOLO detection format: cls cx cy w h
                cls, cx, cy, w, h = vals
                if w > 0 and h > 0:
                    labels.append([cls, cx, cy, w, h])

            elif len(vals) > 5 and (len(vals) - 1) % 2 == 0:
                # YOLO segmentation format: cls x1 y1 x2 y2 ...
                # Convert polygon → axis-aligned bounding box (AABB)
                cls    = vals[0]
                coords = vals[1:]
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                # Compute center coordinates and dimensions
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w  = x_max - x_min
                h  = y_max - y_min
                
                if w > 0 and h > 0:
                    labels.append([cls, cx, cy, w, h])

    if labels:
        return torch.tensor(labels, dtype=torch.float32)
    
    # Return empty tensor if no valid labels found
    return torch.zeros((0, 5), dtype=torch.float32)
