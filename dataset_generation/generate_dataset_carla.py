from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import random
import shutil

from tqdm import tqdm


DATASET_DIR = "YOUR PATH"
OUT_DIR     = "YOUR PATH 1"

# Split ratio: 80% train, 10% validation, 10% test
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description='Split CARLA dataset into YOLO train/val/test sets')
    parser.add_argument('--datadir', default=DATASET_DIR, type=str, help='Source data directory')
    parser.add_argument('--outdir',  default=OUT_DIR,     type=str, help='Output directory')
    return parser.parse_args()


def split_carla(data_dir, out_dir):

    rgb_dir   = os.path.join(data_dir, 'rgb')
    radar_dir = os.path.join(data_dir, 'radar_png')
    label_dir = os.path.join(data_dir, 'semantic_yolo')

    # Collect all basenames that have all 3 required files (RGB, Radar, Label)
    all_basenames = []
    for fname in os.listdir(rgb_dir):
        if not fname.endswith('.png'):
            continue
        base       = os.path.splitext(fname)[0]
        rgb_path   = os.path.join(rgb_dir,   base + '.png')
        radar_path = os.path.join(radar_dir, base + '.png')
        label_path = os.path.join(label_dir, base + '.txt')

        if os.path.isfile(rgb_path) and os.path.isfile(radar_path) and os.path.isfile(label_path):
            all_basenames.append(base)
        else:
            print("Incomplete triplet, skipping: %s" % base)

    print("Total complete files found: %d" % len(all_basenames))

    # Reproducible random shuffle
    random.seed(SEED)
    random.shuffle(all_basenames)

    # Calculate split indices
    n       = len(all_basenames)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_files = all_basenames[:n_train]
    val_files   = all_basenames[n_train:n_train + n_val]
    test_files  = all_basenames[n_train + n_val:]

    print("Train: %d | Val: %d | Test: %d" % (len(train_files), len(val_files), len(test_files)))

    sets      = ['train',      'val',      'test']
    all_files = [train_files,  val_files,  test_files]

    for data_set, file_list in zip(sets, all_files):

        # Create output directories
        cam_out   = os.path.join(out_dir, 'images',       data_set)
        radar_out = os.path.join(out_dir, 'images_radar', data_set)
        label_out = os.path.join(out_dir, 'labels',       data_set)
        os.makedirs(cam_out,   exist_ok=True)
        os.makedirs(radar_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)

        print('Copying %s set...' % data_set)
        for base in tqdm(file_list):
            shutil.copy2(os.path.join(rgb_dir,   base + '.png'), os.path.join(cam_out,   base + '.png'))
            shutil.copy2(os.path.join(radar_dir, base + '.png'), os.path.join(radar_out, base + '.png'))
            shutil.copy2(os.path.join(label_dir, base + '.txt'), os.path.join(label_out, base + '.txt'))

    # Generate the data.yaml file for YOLO
    yaml_content = (
        "path: %s\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n\n"
        "nc: 1\n"
        "names: ['vehicle']\n"
    ) % out_dir

    with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    print("Finished. Final directory structure:")
    print("  %s/images/train|val|test        -> Camera images" % out_dir)
    print("  %s/images_radar/train|val|test  -> Radar images"  % out_dir)
    print("  %s/labels/train|val|test        -> YOLO labels"   % out_dir)
    print("  %s/data.yaml"                                     % out_dir)


if __name__ == '__main__':
    args = parse_args()
    split_carla(args.datadir, args.outdir)
