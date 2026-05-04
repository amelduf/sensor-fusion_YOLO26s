import carla
import cv2
import numpy as np
import math
import h5py
import queue
import os

# ==========================================================
# DIRECTORY SETUP
# ==========================================================
# Create a structured dataset hierarchy
folders = ["dataset/rgb", "dataset/semantic", "dataset/radar", "dataset/vehicle"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Queues to hold sensor data for synchronous processing
rgb_queue = queue.Queue()
sem_queue = queue.Queue()
radar_queue = queue.Queue()

# ==========================================================
# CARLA WORLD & WEATHER SETUP
# ==========================================================
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Define atmospheric conditions (Foggy Scenario)
foggy = carla.WeatherParameters(
    cloudiness=60.0,
    fog_density=60.0,
    fog_distance=20.0,
    sun_altitude_angle=15.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wetness=0.0
)
world.set_weather(foggy)

# Configure Synchronous Mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS (1 / 20)
world.apply_settings(settings)

SIM_TIME = 100.0  # Total simulation duration in seconds
DT = settings.fixed_delta_seconds
N_TICKS = int(SIM_TIME / DT)

# Synchronize Traffic Manager
tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)

# ==========================================================
# ACTOR SPAWNING (Ego-Vehicle)
# ==========================================================
spawn_points = world.get_map().get_spawn_points()
vehicle_bp = world.get_blueprint_library().filter('*vehicle*')

vehicle = None
for spawn in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp[0], spawn)
    if vehicle is not None:
        break

if vehicle is None:
    raise RuntimeError("Could not find a free spawn point.")

vehicle.set_autopilot(True)

# ==========================================================
# SENSOR CONFIGURATION
# ==========================================================
# Sensor positioning offsets (relative to vehicle center)
CAMERA_POS_Z = 1.6  # Height
CAMERA_POS_X = 0.9  # Forward

# 1. Semantic Segmentation Camera
sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
sem_bp.set_attribute('image_size_x', '1600')
sem_bp.set_attribute('image_size_y', '900')
sem_bp.set_attribute('sensor_tick', '0.0')
sem_tf = carla.Transform(carla.Location(z=CAMERA_POS_Z, x=CAMERA_POS_X))
camera_sem = world.spawn_actor(sem_bp, sem_tf, attach_to=vehicle)

# 2. RGB Camera
rgb_bp = world.get_blueprint_library().find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', '1600')
rgb_bp.set_attribute('image_size_y', '900')
rgb_bp.set_attribute('sensor_tick', '0.0')
rgb_tf = carla.Transform(carla.Location(z=CAMERA_POS_Z, x=CAMERA_POS_X))
camera_rgb = world.spawn_actor(rgb_bp, rgb_tf, attach_to=vehicle)

# 3. Radar Sensor
radar_bp = world.get_blueprint_library().find('sensor.other.radar')
radar_bp.set_attribute('horizontal_fov', '90')
radar_bp.set_attribute('vertical_fov', '1')
radar_bp.set_attribute('range', '100')
radar_bp.set_attribute('sensor_tick', '0.0')
radar_bp.set_attribute('points_per_second', '1000')
radar_tf = carla.Transform(carla.Location(x=1.0, z=0.8), carla.Rotation(pitch=0))
radar = world.spawn_actor(radar_bp, radar_tf, attach_to=vehicle)

# Connect sensors to their respective queues
camera_rgb.listen(rgb_queue.put)
camera_sem.listen(sem_queue.put)
radar.listen(radar_queue.put)

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def save_vehicle_state(vehicle, frame_id):
    """Logs the ground truth state of the ego-vehicle."""
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()

    data = {
        "position": [transform.location.x, transform.location.y, transform.location.z],
        "velocity": [velocity.x, velocity.y, velocity.z],
        "yaw": transform.rotation.yaw
    }

    filename = f"dataset/vehicle/t10_005_{frame_id:06d}.h5"
    with h5py.File(filename, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)

# ==========================================================
# DATA ACQUISITION LOOP
# ==========================================================
print(f"Starting simulation for {N_TICKS} ticks...")

try:
    for i in range(N_TICKS):
        # Step the simulation
        frame_id = world.tick()

        # Retrieve data from queues
        # .get() blocks until data is available
        rgb_raw = rgb_queue.get()
        sem_raw = sem_queue.get()
        radar_raw = radar_queue.get()

        # Hard synchronization check
        if not (rgb_raw.frame == sem_raw.frame == radar_raw.frame):
            print(f"[Warning] Frame mismatch @ {frame_id}. Skipping...")
            continue

        # 1. Process RGB Image
        rgb_arr = np.frombuffer(rgb_raw.raw_data, dtype=np.uint8)
        rgb_arr = rgb_arr.reshape((rgb_raw.height, rgb_raw.width, 4))
        rgb_final = rgb_arr[:, :, :3] # Remove alpha channel

        # 2. Process Semantic Segmentation
        # In CARLA, the red channel contains the raw Category ID
        sem_arr = np.frombuffer(sem_raw.raw_data, dtype=np.uint8)
        sem_arr = sem_arr.reshape((sem_raw.height, sem_raw.width, 4))
        sem_labels = sem_arr[:, :, 2]

        # 3. Process Radar Detections
        # Data format: [depth, velocity, azimuth, altitude]
        radar_dets = [[d.depth, d.velocity, d.azimuth, d.altitude] for d in radar_raw]
        radar_array = np.array(radar_dets, dtype=np.float32)

        # 4. Save Multimodal Data
        # Save RGB as PNG
        cv2.imwrite(f"dataset/rgb/t10_005_{frame_id:06d}.png", rgb_final)

        # Save Semantic Labels (HDF5 with Compression)
        with h5py.File(f"dataset/semantic/t10_005_{frame_id:06d}.h5", "w") as f:
            f.create_dataset("labels", data=sem_labels, compression="gzip")

        # Save Radar Data (HDF5)
        with h5py.File(f"dataset/radar/t10_005_{frame_id:06d}.h5", "w") as f:
            f.create_dataset("radar", data=radar_array)

        # Save Ego-Vehicle GT
        save_vehicle_state(vehicle, frame_id)

        if i % 100 == 0:
            print(f"Progress: {i}/{N_TICKS} frames collected.")

finally:
    # ==========================================================
    # CLEANUP
    # ==========================================================
    print("Stopping sensors and cleaning up actors...")
    camera_rgb.stop()
    camera_sem.stop()
    radar.stop()

    # Disable synchronous mode before exiting
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    # Destroy all spawned actors
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()
    for sensor in world.get_actors().filter('*sensor*'):
        sensor.destroy()

    print("Dataset generation complete.")
