import os
import numpy as np
from PIL import Image
from tqdm import tqdm

radar_dir = '/YOUR PATH/images_radar/train'

mean = np.zeros(3)
std  = np.zeros(3)
n    = 0

files = [f for f in os.listdir(radar_dir) if f.endswith('.png')]

for fname in tqdm(files):
    img = np.array(Image.open(os.path.join(radar_dir, fname)).convert('RGB')).astype(np.float32) / 255.0
    mean += img.mean(axis=(0,1))
    std  += img.std(axis=(0,1))
    n    += 1

mean /= n
std  /= n

print("radar_mean: [%.4f, %.4f, %.4f]" % (mean[0], mean[1], mean[2]))
print("radar_std:  [%.4f, %.4f, %.4f]" % (std[0],  std[1],  std[2]))
