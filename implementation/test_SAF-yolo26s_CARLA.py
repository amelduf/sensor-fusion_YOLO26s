#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAF-YOLO26s — Real-Time Inference in CARLA
=================================================
Radar projection logic matches the dataset generation script:
  - Projection on 1600x900 with circle radius=7 + z-buffer
  - Resize 1600x900 → 640x640
  - Identical color encoding: R=depth, G=velocity, B=128

Dependencies:
  pip install torch ultralytics pyyaml opencv-python numpy h5py filterpy scipy
"""

import time
import math
import queue
import numpy as np
import cv2
import yaml
import torch
import carla
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ══════════════════════════════════════════════════════
# SORT TRACKER (Kalman Filter + IoU matching)
# ══════════════════════════════════════════════════════

def iou_batch(bb_test, bb_gt):
    """
    Computes Intersection over Union (IoU) between two batches of bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    bb_gt   = np.expand_dims(bb_gt,   0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w   = np.maximum(0., xx2 - xx1)
    h   = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area2 = (bb_gt[...,  2] - bb_gt[...,  0]) * (bb_gt[...,  3] - bb_gt[...,  1])
    return inter / (area1 + area2 - inter + 1e-6)


def convert_bbox_to_z(bbox):
    """
    Converts [x1,y1,x2,y2] to Kalman state format [x, y, s, r]
    where s is scale (area) and r is aspect ratio.
    """
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.;  y = bbox[1] + h / 2.
    s = w * h; r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Converts Kalman state [x,y,s,r] back to [x1,y1,x2,y2].
    """
    w = np.sqrt(x[2] * x[3]); h = x[2] / w
    b = [x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]
    return np.array(b) if score is None else np.array([*b, score])


class KalmanBoxTracker:
    """
    Extended Kalman Filter: 
    State = [x, y, s, r, vx, vy, vs, dist, v_dist]
    dist: distance to vehicle (from radar or estimated via ratio)
    v_dist: approach velocity
    """
    count = 0

    def __init__(self, bbox, dist=None):
        # dim_x=9 (state), dim_z=5 (observations: x,y,s,r,dist)
        self.kf = KalmanFilter(dim_x=9, dim_z=5)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0,0], # x
            [0,1,0,0,0,1,0,0,0], # y
            [0,0,1,0,0,0,1,0,0], # s
            [1,0,0,1,0,0,0,0,0], # r (static)
            [0,0,0,0,1,0,0,0,0], # vx
            [0,0,0,0,0,1,0,0,0], # vy
            [0,0,0,0,0,0,1,0,0], # vs
            [0,0,0,0,0,0,0,1,1], # dist
            [0,0,0,0,0,0,0,0,1], # v_dist
        ], dtype=float)
        
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0],
        ], dtype=float)
        
        # Measurement noise matrix
        self.kf.R = np.diag([1., 1., 10., 1., 2.]).astype(float)
        self.kf.P[4:7, 4:7] *= 1000. 
        self.kf.P           *= 10.
        self.kf.P[7, 7]      = 100.
        self.kf.P[8, 8]      = 10.
        
        # Process noise matrix
        self.kf.Q[-1, -1]   *= 0.01
        self.kf.Q[4:7, 4:7] *= 0.01
        self.kf.Q[7, 7]      = 0.5
        self.kf.Q[8, 8]      = 0.1
        
        z = convert_bbox_to_z(bbox)
        d0 = dist if dist is not None else 50.0
        self.kf.x[:4] = z
        self.kf.x[7]  = d0
        self.kf.x[8]  = 0.0
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hit_streak = 0
        self.conf = bbox[4] if len(bbox) > 4 else 1.0
        self.dist_method = "radar" if dist is not None else "ratio"

    def update(self, bbox, dist=None):
        """Updates the state with a new measurement."""
        self.time_since_update = 0
        self.hit_streak += 1
        self.conf = bbox[4] if len(bbox) > 4 else self.conf
        z = convert_bbox_to_z(bbox)
        
        d = dist if dist is not None else float(self.kf.x[7])
        self.dist_method = "radar" if dist is not None else "ratio"
        
        # Adjust measurement uncertainty based on source
        self.kf.R[4, 4] = 0.5 if dist is not None else 50.0
        
        obs = np.array([z[0,0], z[1,0], z[2,0], z[3,0], d]).reshape(5, 1)
        self.kf.update(obs)

    def predict(self):
        """Advances the state vector and returns predicted bounding box."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x).reshape(1, 4)

    def get_state(self):
        return convert_x_to_bbox(self.kf.x).reshape(1, 4)

    def get_distance(self):
        return float(self.kf.x[7]), self.dist_method


class SORTTracker:
    def __init__(self, max_age=15, min_hits=1, iou_threshold=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, dists=None):
        """
        dets: np.array (N,5) [x1,y1,x2,y2,conf]
        dists: list/array (N,) distances or None
        Returns a list of tracked objects: [x1,y1,x2,y2,conf,id,dist,method]
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for i, t in enumerate(self.trackers):
            pos = t.predict().flatten()[:4]
            if len(pos) < 4 or np.any(np.isnan(pos)):
                to_del.append(i)
                continue
            trks[i] = [*pos, t.conf]
            
        for i in reversed(to_del):
            self.trackers.pop(i)
            trks = np.delete(trks, i, axis=0)

        matched, unmatched_dets, unmatched_trks = self._associate(dets, trks)

        # Update matched trackers
        for d, t in matched:
            dist = dists[d] if dists is not None and d < len(dists) else None
            self.trackers[t].update(dets[d], dist=dist)

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            dist = dists[i] if dists is not None and i < len(dists) else None
            self.trackers.append(KalmanBoxTracker(dets[i], dist=dist))

        ret = []
        for t in reversed(range(len(self.trackers))):
            trk = self.trackers[t]
            if trk.time_since_update <= self.max_age and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state().flatten().tolist()
                dist_val, dist_method = trk.get_distance()
                ret.append([*d, trk.conf, trk.id, dist_val, dist_method])
            
            if trk.time_since_update > self.max_age:
                self.trackers.pop(t)
        return ret

    def _associate(self, dets, trks):
        """Assigns detections to tracked objects via Hungarian Algorithm."""
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))
            
        iou_mat = iou_batch(dets[:, :4], trks[:, :4])
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        
        matched, unmatched_dets, unmatched_trks = [], [], []
        for d in range(len(dets)):
            if d not in row_ind:
                unmatched_dets.append(d)
        for t in range(len(trks)):
            if t not in col_ind:
                unmatched_trks.append(t)
                
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] < self.iou_threshold:
                unmatched_dets.append(r)
                unmatched_trks.append(c)
            else:
                matched.append([r, c])
        return matched, unmatched_dets, unmatched_trks

# ══════════════════════════════════════════════════════
# SYSTEM CONFIGURATION
# ══════════════════════════════════════════════════════
HOST = "localhost"
PORT = 2000

SIM_FPS       = 60.0
SIM_TIME_SEC  = 100.0

IMAGE_W    = 1600
IMAGE_H    = 900
CAMERA_FOV = 90.0

IMG_SIZE   = 640
CONF_THRES = 0.05

CFG_PATH     = "SAF_YOLO26.yaml"
WEIGHTS_PATH = "saf_yolo_sim.pt"
VARIANT      = "yolo26s"

# Extrinsics — Must match dataset generation script
CAMERA_POS_X = 0.9
CAMERA_POS_Z = 1.6
RADAR_POS_X  = 1.0
RADAR_POS_Z  = 0.8
RADAR_PITCH_DEG = 0

# CARLA Radar Attributes
RADAR_HFOV  = "90"
RADAR_VFOV  = "1"
RADAR_RANGE = "100"
RADAR_TICK  = "0.0"
RADAR_PPS   = "1000"

RADAR_RADIUS = 7 # Circle radius in pixel rendering

# Display settings
WINDOW_NAME  = "SAF-YOLO26s — CARLA Inference"
SHOW_OVERLAY = False # Draw radar points on raw image

# Video Recording
SAVE_VIDEO   = False
VIDEO_PATH   = "saf_carla_demo.mp4"
VIDEO_FPS    = 20

# Debug settings (Costly — Disable for max FPS)
DRAW_WORLD_DEBUG    = False
WORLD_LIFE_TIME     = 0.05
MAX_WORLD_RADAR_PTS = 200

# ══════════════════════════════════════════════════════
# CAMERA INTRINSICS
# ══════════════════════════════════════════════════════
CAMERA_POS = np.array([CAMERA_POS_X, 0.0, CAMERA_POS_Z], dtype=np.float32)
RADAR_POS  = np.array([RADAR_POS_X,  0.0, RADAR_POS_Z],  dtype=np.float32)

def get_camera_intrinsic(w, h, fov_deg):
    fov_rad = np.deg2rad(fov_deg)
    fx = w / (2.0 * np.tan(fov_rad / 2.0))
    return np.array([[fx, 0,  w / 2.0],
                     [0,  fx, h / 2.0],
                     [0,  0,  1.0    ]], dtype=np.float32)

K_FULL = get_camera_intrinsic(IMAGE_W, IMAGE_H, CAMERA_FOV)

def rotation_matrix_pitch(pitch_deg):
    p = np.deg2rad(pitch_deg)
    return np.array([
        [ np.cos(p), 0.0, np.sin(p)],
        [ 0.0,       1.0, 0.0      ],
        [-np.sin(p), 0.0, np.cos(p)]
    ], dtype=np.float32)

R_RADAR = rotation_matrix_pitch(RADAR_PITCH_DEG)

# ══════════════════════════════════════════════════════
# PROJECTION RADAR → UV (1600x900)
# ══════════════════════════════════════════════════════
def project_radar_to_image_full(radar_meas):
    """
    Converts CARLA radar data to UV coordinates (1600x900).
    Logic identical to the dataset generation script.
    """
    if len(radar_meas) == 0:
        return None, None, None

    dets = np.empty((len(radar_meas), 4), dtype=np.float32)
    for i, d in enumerate(radar_meas):
        dets[i, 0] = d.depth
        dets[i, 1] = d.velocity
        dets[i, 2] = d.azimuth
        dets[i, 3] = d.altitude

    depth0 = dets[:, 0]
    az     = dets[:, 2]
    alt    = dets[:, 3]

    # Spherical → Cartesian (Radar frame)
    x = depth0 * np.cos(alt) * np.cos(az)
    y = depth0 * np.cos(alt) * np.sin(az)
    z = depth0 * np.sin(alt)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    # Apply Radar pitch rotation
    pts = (R_RADAR @ pts.T).T

    # Translate from Radar frame to Camera frame
    pts[:, 0] += RADAR_POS[0] - CAMERA_POS[0]
    pts[:, 1] += RADAR_POS[1] - CAMERA_POS[1]
    pts[:, 2] += RADAR_POS[2] - CAMERA_POS[2]

    # Sensor frame → Camera Coordinate System (CARLA convention)
    pts_cam = np.empty_like(pts)
    pts_cam[:, 0] =  pts[:, 1]   # X_cam =  Y_radar
    pts_cam[:, 1] = -pts[:, 2]   # Y_cam = -Z_radar
    pts_cam[:, 2] =  pts[:, 0]   # Z_cam =  X_radar

    # Filter points in front of the camera
    mask    = pts_cam[:, 2] > 0.0
    pts_cam = pts_cam[mask]
    depth   = pts_cam[:, 2].copy()
    vel     = dets[:, 1][mask]

    if pts_cam.shape[0] == 0:
        return None, None, None

    # Project to 1600x900 plane
    uvw = (K_FULL @ pts_cam.T).T
    uv  = uvw[:, :2] / uvw[:, 2:3]

    valid = (
        (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_H)
    )
    return uv[valid].astype(int), depth[valid], vel[valid]


# ══════════════════════════════════════════════════════
# RADAR IMAGE GENERATION (Circles + Z-buffer)
# ══════════════════════════════════════════════════════
def make_radar_image(uv, depth, velocity, radius=RADAR_RADIUS):
    """
    Renders the 1600x900 radar image using circle primitives and z-buffering.
    Resizes to 640x640 to match model input.
    """
    img          = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    depth_buffer = np.full((IMAGE_H, IMAGE_W), np.inf)

    if uv is None or len(uv) == 0:
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    for i in range(len(uv)):
        u, v = uv[i]
        d    = depth[i]
        vel  = velocity[i]

        R = int(np.clip((128.0 * d   / 100.0) + 127.0, 0, 255))
        G = int(np.clip((128.0 * (vel + 20.0) / 40.0) + 127.0, 0, 255))
        B = 128

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                px, py = u + dx, v + dy
                if px < 0 or px >= IMAGE_W or py < 0 or py >= IMAGE_H:
                    continue
                # Z-buffer rule
                if d < depth_buffer[py, px]:
                    depth_buffer[py, px] = d
                    img[py, px] = (B, G, R)

    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════
# SAF MODEL LOADING
# ══════════════════════════════════════════════════════
def load_saf_model(cfg_path, weights_path, variant):
    from SAF_YOLO26 import SAFYolo26

    cfg     = yaml.safe_load(open(cfg_path, "r"))
    norm    = cfg["normalization"]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    model = SAFYolo26(variant=variant, pretrained=False).to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    use_fp16 = (device.type == "cuda")
    if use_fp16:
        model.half()

    # Normalization parameters from config
    cam_mean = np.array(norm.get("cam_mean",   [0.485, 0.456, 0.406]), dtype=np.float32)
    cam_std  = np.array(norm.get("cam_std",    [0.229, 0.224, 0.225]), dtype=np.float32)
    rad_mean = np.array(norm.get("radar_mean", [0.003, 0.003, 0.003]), dtype=np.float32)
    rad_std  = np.array(norm.get("radar_std",  [0.047, 0.045, 0.038]), dtype=np.float32)

    print(f"Model loaded — Epoch {ckpt.get('epoch', '?')}")
    return model, device, use_fp16, cam_mean, cam_std, rad_mean, rad_std


def to_tensor(img_rgb, mean, std, device, fp16=False):
    """Normalizes and converts RGB image to PyTorch tensor (1xCxHxW)."""
    x = torch.from_numpy(img_rgb).to(device)
    x = x.permute(2, 0, 1).float().div_(255.0)
    m = torch.as_tensor(mean, device=device).view(3, 1, 1)
    s = torch.as_tensor(std,  device=device).view(3, 1, 1)
    x = (x - m) / s
    return x.unsqueeze(0).half() if fp16 else x.unsqueeze(0)


# ══════════════════════════════════════════════════════
# DISTANCE ESTIMATION LOGIC
# ══════════════════════════════════════════════════════
VEHICLE_HEIGHT_M = 1.5  # Average vehicle height fallback

def get_focal_length():
    """Focal length in pixels for the current setup."""
    fov_rad = np.deg2rad(CAMERA_FOV)
    return IMAGE_W / (2.0 * np.tan(fov_rad / 2.0))

FX = get_focal_length()

def estimate_distance(bbox_img, uv_full, depth_full):
    """
    Estimates distance for a detected object.
    Method 1: Direct radar projection lookup inside the bounding box.
    Method 2: Fallback using the geometry of the bbox height (focal ratio).
    """
    x1, y1, x2, y2 = bbox_img

    # 1. Primary Method: Radar point within bbox
    if uv_full is not None and len(uv_full) > 0:
        in_box = (
            (uv_full[:, 0] >= x1) & (uv_full[:, 0] <= x2) &
            (uv_full[:, 1] >= y1) & (uv_full[:, 1] <= y2)
        )
        if np.any(in_box):
            return float(np.min(depth_full[in_box])), 'radar'

    # 2. Fallback: Geometric estimation
    h_px = max(1, y2 - y1)
    dist = (FX * VEHICLE_HEIGHT_M) / h_px
    return dist, 'ratio'


# ══════════════════════════════════════════════════════
# VISUALIZATION UTILITIES
# ══════════════════════════════════════════════════════
def draw_radar_overlay(rgb_full, uv, depth, vel, radius=RADAR_RADIUS):
    """Overlays projected radar detections on the full-res camera image."""
    out = rgb_full.copy()
    if uv is None or len(uv) == 0:
        return out

    for i in range(len(uv)):
        u, v_  = uv[i]
        d      = depth[i]
        veloc  = vel[i]
        R = int(np.clip((128.0 * d       / 100.0) + 127.0, 0, 255))
        G = int(np.clip((128.0 * (veloc + 20.0) / 40.0) + 127.0, 0, 255))
        cv2.circle(out, (u, v_), radius, (128, G, R), -1)

    return out


def draw_detections(img, tracked):
    """Draws tracked bounding boxes with ID and Kalman-filtered distance."""
    H, W = img.shape[:2]
    out  = img.copy()
    sx, sy = W / IMG_SIZE, H / IMG_SIZE

    colors = [(0,255,0),(0,200,255),(255,100,0),(255,0,200),(100,255,100),
              (0,100,255),(255,255,0),(200,0,255),(0,255,200),(255,150,50)]

    for det in tracked:
        if len(det) < 6: continue
        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
        conf, tid, dist, meth = det[4], int(det[5]), det[6], det[7]

        x1i, y1i = int(np.clip(x1 * sx, 0, W - 1)), int(np.clip(y1 * sy, 0, H - 1))
        x2i, y2i = int(np.clip(x2 * sx, 0, W - 1)), int(np.clip(y2 * sy, 0, H - 1))
        
        if x2i <= x1i or y2i <= y1i: continue

        color = colors[tid % len(colors)]
        icon  = 'R' if meth == 'radar' else '~'
        label = f"ID:{tid} {icon}{dist:.1f}m"

        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1i, max(0, y1i-th-8)), (x1i+tw+4, y1i), color, -1)
        cv2.putText(out, label, (x1i+2, max(0, y1i-4)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out


def draw_world_debug(world, vehicle, dets_np, radar_meas, radar_actor):
    """Draws detections and radar points directly in the CARLA 3D world."""
    n    = len(dets_np)
    loc  = vehicle.get_location() + carla.Location(z=2.2)
    conf = float(np.max(dets_np[:, 4])) if n > 0 else 0.0
    world.debug.draw_string(loc, f"SAF dets={n} conf={conf:.2f}", 
                            draw_shadow=True, color=carla.Color(255, 255, 0), 
                            life_time=WORLD_LIFE_TIME)

    tf   = radar_actor.get_transform()
    pts  = []
    for d in radar_meas:
        r, az, al = d.depth, d.azimuth, d.altitude
        lw = tf.transform(carla.Location(
            x=float(r * math.cos(al) * math.cos(az)),
            y=float(r * math.cos(al) * math.sin(az)),
            z=float(r * math.sin(al))))
        pts.append((lw, float(d.velocity)))

    step = max(1, len(pts) // MAX_WORLD_RADAR_PTS)
    for loc_w, vel in pts[::step]:
        t = max(0.0, min(1.0, (vel + 20.0) / 40.0))
        world.debug.draw_point(loc_w, size=0.06, 
                               color=carla.Color(int(255*t), int(255*(1-t)), 50), 
                               life_time=WORLD_LIFE_TIME)


# ══════════════════════════════════════════════════════
# MAIN EXECUTION LOOP
# ══════════════════════════════════════════════════════
def main():
    model, device, use_fp16, cam_mean, cam_std, rad_mean, rad_std = load_saf_model(
        CFG_PATH, WEIGHTS_PATH, VARIANT)

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()

    # Synchronous mode settings
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    spawn_points = world.get_map().get_spawn_points()
    actors_to_destroy = []
    rgb_queue = queue.Queue()
    radar_queue = queue.Queue()

    try:
        # ── Setup Ego Vehicle ──
        bp_lib  = world.get_blueprint_library()
        veh_bps = bp_lib.filter("*vehicle*")
        vehicle = None
        for sp in spawn_points:
            vehicle = world.try_spawn_actor(veh_bps[0], sp)
            if vehicle: break
        if vehicle is None: raise RuntimeError("No spawn points available")
        actors_to_destroy.append(vehicle)
        vehicle.set_autopilot(True)

        # ── Setup RGB Camera ──
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        cam_tf  = carla.Transform(carla.Location(x=CAMERA_POS_X, z=CAMERA_POS_Z))
        cam_rgb = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        actors_to_destroy.append(cam_rgb)

        # ── Setup Radar ──
        rad_bp = bp_lib.find("sensor.other.radar")
        rad_bp.set_attribute("horizontal_fov",    RADAR_HFOV)
        rad_bp.set_attribute("vertical_fov",      RADAR_VFOV)
        rad_bp.set_attribute("range",             RADAR_RANGE)
        rad_bp.set_attribute("points_per_second", RADAR_PPS)
        rad_tf = carla.Transform(carla.Location(x=RADAR_POS_X, z=RADAR_POS_Z),
                                 carla.Rotation(pitch=RADAR_PITCH_DEG))
        radar_actor = world.spawn_actor(rad_bp, rad_tf, attach_to=vehicle)
        actors_to_destroy.append(radar_actor)

        cam_rgb.listen(rgb_queue.put)
        radar_actor.listen(radar_queue.put)

        # ── Warmup ──
        for _ in range(10):
            world.tick()
            if not rgb_queue.empty():   rgb_queue.get()
            if not radar_queue.empty(): radar_queue.get()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

        video_writer = None
        if SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (IMAGE_W, IMAGE_H))

        tracker = SORTTracker(max_age=30, min_hits=1, iou_threshold=0.05)

        n_ticks = int(SIM_TIME_SEC / settings.fixed_delta_seconds)
        fps_counter, fps_timer, fps_display = 0, time.time(), 0

        print("\nSimulation Started — Press 'q' to exit\n")

        for _ in range(n_ticks):
            world.tick()

            # FPS logic
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display, fps_counter, fps_timer = fps_counter, 0, time.time()

            # Retrieve data from queues
            rgb_frame = rgb_queue.get()
            while not rgb_queue.empty(): rgb_frame = rgb_queue.get()
            radar_meas = radar_queue.get()
            while not radar_queue.empty(): radar_meas = radar_queue.get()

            # Process Camera Frame
            arr = np.frombuffer(rgb_frame.raw_data, dtype=np.uint8)
            arr = arr.reshape((rgb_frame.height, rgb_frame.width, 4))
            rgb_full = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
            cam_640  = cv2.resize(rgb_full, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

            # Process Radar Frame (Match dataset encoding)
            uv, depth, vel = project_radar_to_image_full(radar_meas)
            radar_640      = make_radar_image(uv, depth, vel)

            # Prepare Tensors for Inference
            cam_t = to_tensor(cam_640,   cam_mean, cam_std, device, use_fp16)
            rad_t = to_tensor(radar_640, rad_mean, rad_std, device, use_fp16)

            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.time()

            # ── Model Inference ──
            with torch.inference_mode():
                preds = model(rad_t, cam_t)

            if device.type == "cuda": torch.cuda.synchronize()
            t_infer = (time.time() - t0) * 1000 # ms latency

            dets = preds[0][0].detach()
            if device.type == "cuda": dets = dets.float()
            dets_np = dets[dets[:, 4] >= CONF_THRES].cpu().numpy()

            # Estimate distance for tracking
            dists = []
            if len(dets_np) > 0:
                for det in dets_np:
                    sx_, sy_ = IMAGE_W / IMG_SIZE, IMAGE_H / IMG_SIZE
                    bbox_img = [int(det[0]*sx_), int(det[1]*sy_), int(det[2]*sx_), int(det[3]*sy_)]
                    d, _ = estimate_distance(bbox_img, uv, depth)
                    dists.append(d)

            # Update Tracker
            tracked = tracker.update(
                dets_np[:, :5] if len(dets_np) > 0 else np.empty((0, 5)),
                dists=dists if dists else None
            )

            if DRAW_WORLD_DEBUG:
                draw_world_debug(world, vehicle, dets_np, radar_meas, radar_actor)

            # ── Visualization ──
            vis = draw_radar_overlay(rgb_full, uv, depth, vel) if SHOW_OVERLAY else rgb_full
            vis = draw_detections(vis, tracked)

            cv2.putText(vis, f"FPS:{fps_display}  Latency:{t_infer:.1f}ms  Tracks:{len(tracked)}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            if SAVE_VIDEO and video_writer:
                video_writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord("q"): break

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if SAVE_VIDEO and video_writer: video_writer.release()
        for a in actors_to_destroy:
            try: a.destroy()
            except: pass
        # Revert CARLA settings
        try:
            s = world.get_settings()
            s.synchronous_mode, s.fixed_delta_seconds = False, None
            world.apply_settings(s)
            tm.set_synchronous_mode(False)
        except: pass
        print("Simulation ended.")

if __name__ == "__main__":
    main()
