import numpy as np
import cv2
import h5py
import os
import math

# ==========================================================
# CAMERA / RADAR CALIBRATION & CONFIGURATION
# ==========================================================
IMAGE_W = 1600
IMAGE_H = 900
CAMERA_FOV = 90.0  # Default CARLA Camera FOV

# Extrinsic parameters: Position [X, Y, Z] in meters (CARLA coordinates)
CAMERA_POS = np.array([0.9, 0.0, 1.6])
RADAR_POS  = np.array([1.0, 0.0, 0.8])
RADAR_PITCH_DEG = 0  # Radar tilt angle

# ==========================================================
# GEOMETRY UTILITIES
# ==========================================================

def get_camera_intrinsic(w, h, fov_deg):
    """
    Computes the Camera Intrinsic Matrix (K) based on FOV and image dimensions.
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = w / (2 * np.tan(fov_rad / 2))
    fy = fx

    return np.array([
        [fx, 0, w / 2],
        [0, fy, h / 2],
        [0,  0,     1]
    ])


def radar_to_cartesian(radar):
    """
    Converts Radar spherical detections (Range, Velocity, Azimuth, Altitude) 
    to Cartesian coordinates (X, Y, Z) in the Radar's local frame.
    """
    depth = radar[:, 0]
    az = radar[:, 2]
    alt = radar[:, 3]

    x = depth * np.cos(alt) * np.cos(az)
    y = depth * np.cos(alt) * np.sin(az)
    z = depth * np.sin(alt)

    return np.stack([x, y, z], axis=1)


def rotation_matrix(pitch_deg):
    """
    Creates a 3D rotation matrix for the pitch axis.
    """
    pitch = np.deg2rad(pitch_deg)
    return np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,               1, 0            ],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])


# ==========================================================
# PROJECTION PIPELINE: RADAR -> IMAGE PLANE
# ==========================================================

def project_radar_to_image(radar_h5):
    """
    Reads radar data and projects 3D points onto the 2D camera image plane.
    """
    with h5py.File(radar_h5, 'r') as f:
        radar = f['radar'][:]

    if radar.shape[0] == 0:
        return None, None, None

    # 1. Convert to local Cartesian coordinates
    pts = radar_to_cartesian(radar)

    # 2. Apply Radar pitch rotation
    pts = (rotation_matrix(RADAR_PITCH_DEG) @ pts.T).T

    # 3. Translate from Radar frame to World frame, then to Camera frame
    # Adjusting for the relative offset between sensor positions
    pts[:, 0] += (RADAR_POS[0] - CAMERA_POS[0])
    pts[:, 1] += (RADAR_POS[1] - CAMERA_POS[1])
    pts[:, 2] += (RADAR_POS[2] - CAMERA_POS[2])

    # 4. Transform to Camera Coordinate System (UE4/CARLA Convention)
    # Forward(X), Right(Y), Up(Z) -> Right(X), Down(Y), Forward(Z)
    pts_cam = np.zeros_like(pts)
    pts_cam[:, 0] =  pts[:, 1]   # Camera X is Radar Y
    pts_cam[:, 1] = -pts[:, 2]   # Camera Y is -Radar Z
    pts_cam[:, 2] =  pts[:, 0]   # Camera Z is Radar X (Depth)

    # 5. Filter points behind the camera
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    depth = pts_cam[:, 2]
    velocity = radar[:, 1][mask] # Keep velocity for visualization

    # 6. Project 3D points to 2D Pixel coordinates using Intrinsics
    K = get_camera_intrinsic(IMAGE_W, IMAGE_H, CAMERA_FOV)
    uv = (K @ pts_cam.T).T
    uv[:, 0] /= uv[:, 2]
    uv[:, 1] /= uv[:, 2]

    # 7. Final filtering: Keep only points within the image boundaries
    valid = (
        (uv[:, 0] >= 0) & (uv[:, 0] < IMAGE_W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < IMAGE_H)
    )

    return uv[valid, :2].astype(int), depth[valid], velocity[valid]


# ==========================================================
# VISUALIZATION
# ==========================================================

def draw_radar_image(uv, depth, velocity, save_path, radius=7):
    """
    Renders the projected radar points as a 2D image.
    Uses a Z-buffer logic to ensure closer objects overlap distant ones.
    """
    img = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    depth_buffer = np.full((IMAGE_H, IMAGE_W), np.inf)

    for (u, v), d, vel in zip(uv, depth, velocity):
        # Map depth and velocity to color channels for visualization
        R = int(np.clip((128 * d / 100) + 127, 0, 255))
        G = int(np.clip((128 * (vel + 20) / 40) + 127, 0, 255))
        B = 128  # Constant value for blue channel

        # Draw a circular point for each detection
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy > radius*radius:
                    continue

                x, y = u + dx, v + dy

                # Boundary check
                if 0 <= x < IMAGE_W and 0 <= y < IMAGE_H:
                    # Z-buffer check: only draw if this point is closer than the previous one
                    if d < depth_buffer[y, x]:
                        depth_buffer[y, x] = d
                        img[y, x] = (B, G, R) # OpenCV uses BGR

    cv2.imwrite(save_path, img)

# ==========================================================
# MAIN EXECUTION LOOP
# ==========================================================

if __name__ == "__main__":
    RADAR_DIR = "dataset/radar"
    OUT_DIR = "dataset/radar_png"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Starting Radar-to-Camera projection...")

    for fname in sorted(os.listdir(RADAR_DIR)):
        if not fname.endswith(".h5"):
            continue

        radar_path = os.path.join(RADAR_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname.replace(".h5", ".png"))

        # Process the file
        uv, depth, velocity = project_radar_to_image(radar_path)
        
        if uv is not None:
            draw_radar_image(uv, depth, velocity, out_path)
            print(f"[SUCCESS] Saved: {out_path}")
        else:
            print(f"[SKIP] No valid points in: {fname}")
