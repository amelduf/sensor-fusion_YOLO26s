

import carla
import cv2
import numpy as np
import math
import h5py
import queue
import os

os.makedirs("dataset/rgb", exist_ok=True)
os.makedirs("dataset/semantic", exist_ok=True)
os.makedirs("dataset/radar", exist_ok=True)
os.makedirs("dataset/vehicle", exist_ok=True)

rgb_queue = queue.Queue()
sem_queue = queue.Queue()
radar_queue = queue.Queue()

camera_data = {}
radar_data_buffer = None

# connect to the sim 
client = carla.Client('localhost', 2000)




world = client.get_world()
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
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1./20.
world.apply_settings(settings)
SIM_TIME = 100.0  # secondes
DT = settings.fixed_delta_seconds

N_TICKS = int(SIM_TIME / DT)
# ici: 20 / 0.05 = 400 ticks


tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)

spawn_points = world.get_map().get_spawn_points()

#setting up a car with 2 cameras

#spaw a car and set Autopilot on
vehicle_bp = world.get_blueprint_library().filter('*vehicle*')
vehicle = None
for spawn in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp[0], spawn)
    if vehicle is not None:
        break

if vehicle is None:
    raise RuntimeError("Aucun spawn point libre")

vehicle.set_autopilot(True)


#camera mount offset on the car
CAMERA_POS_Z = 1.6 #1.6m up from the ground
CAMERA_POS_X = 0.9 #0.9m forward

#semantic camera
camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
camera_bp.set_attribute('image_size_x', '1600') # this ratio works in CARLA 9.14 on Windows
camera_bp.set_attribute('image_size_y', '900')
camera_bp.set_attribute('sensor_tick', '0.0')
camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
camera_sem = world.spawn_actor(camera_bp,camera_init_trans,attach_to=vehicle)

#normal rgb camera
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1600') 
camera_bp.set_attribute('image_size_y', '900')
camera_bp.set_attribute('sensor_tick', '0.0')
camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
camera_rgb = world.spawn_actor(camera_bp,camera_init_trans,attach_to=vehicle)

radar_bp = world.get_blueprint_library().find('sensor.other.radar')

radar_bp.set_attribute('horizontal_fov', '90')#60
radar_bp.set_attribute('vertical_fov', '1')#3
radar_bp.set_attribute('range', '100')
radar_bp.set_attribute('sensor_tick', '0.0')
radar_bp.set_attribute('points_per_second', '1000')#8000

radar_transform = carla.Transform(carla.Location(x=1.0, z=0.8),carla.Rotation(pitch=0))

radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)

print("--- Caméra RGB ---")
for k,v in camera_rgb.attributes.items():
    print(k, ":", v)
print(camera_rgb.get_transform())

print("--- Caméra Sémantique ---")
for k,v in camera_sem.attributes.items():
    print(k, ":", v)
print(camera_sem.get_transform())

def sem_callback(image, data_dict):
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    raw = raw.reshape((image.height, image.width, 4))

    labels = raw[:, :, 2].copy()  # IDs CARLA BRUTS
    data_dict['semantic_raw'] = labels


    
def rgb_callback(image, data_dict):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    data_dict['rgb'] = array[:, :, :3].copy()

def radar_callback(radar_data):
    detections = []
    for d in radar_data:
        detections.append([
            d.depth,
            d.velocity,
            d.azimuth,
            d.altitude
        ])
    global radar_data_buffer
    radar_data_buffer = np.array(detections, dtype=np.float32)


def save_vehicle_state(vehicle, frame_id):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()

    data = {
        "position": [transform.location.x,
                     transform.location.y,
                     transform.location.z],
        "velocity": [velocity.x, velocity.y, velocity.z],
        "yaw": transform.rotation.yaw
    }

    with h5py.File(f"dataset/vehicle/t10_005_{frame_id:06d}.h5", "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)


image_w = 1600
image_h = 900

camera_data = {'sem_image': np.zeros((image_h,image_w,4)),
               'rgb_image': np.zeros((image_h,image_w,4))}

# this actually opens a live stream from the cameras
camera_rgb.listen(rgb_queue.put)
camera_sem.listen(sem_queue.put)
radar.listen(radar_queue.put)


for i in range(N_TICKS):

    frame_id = world.tick()

    rgb_image = rgb_queue.get()
    sem_image = sem_queue.get()
    radar_data = radar_queue.get()

    # ---- Synchronisation forte ----
    if not (rgb_image.frame == sem_image.frame == radar_data.frame):
        print("Frame mismatch")
        continue

    # ---- Conversion RGB ----
    rgb = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
    rgb = rgb.reshape((rgb_image.height, rgb_image.width, 4))
    rgb = rgb[:, :, :3]

    # ---- Conversion SEM ----
    raw = np.frombuffer(sem_image.raw_data, dtype=np.uint8)
    raw = raw.reshape((sem_image.height, sem_image.width, 4))
    labels = raw[:, :, 2]

    # ---- Conversion RADAR ----
    detections = []
    for d in radar_data:
        detections.append([d.depth, d.velocity, d.azimuth, d.altitude])
    radar_array = np.array(detections, dtype=np.float32)

    # ---- Sauvegarde ----
    cv2.imwrite(f"dataset/rgb/t10_005_{frame_id:06d}.png", rgb)

    with h5py.File(f"dataset/semantic/t10_005_{frame_id:06d}.h5", "w") as f:
        f.create_dataset("labels", data=labels, compression="gzip")

    with h5py.File(f"dataset/radar/t10_005_{frame_id:06d}.h5", "w") as f:
        f.create_dataset("radar", data=radar_array)

    save_vehicle_state(vehicle, frame_id)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
camera_sem.stop() # this is the opposite of camera.listen
camera_rgb.stop() 

for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()


# clean up if something went wrong

camera_sem.stop()
camera_rgb.stop() 
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()

