from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import shutil

from nuscenes.nuscenes import NuScenes
from tqdm import tqdm


RADIUS = 7


def parse_args():
    parser = argparse.ArgumentParser(description='Copies camera and radar images for YOLO training')
    parser.add_argument('--dataset', default='nuscenes', type=str)
    parser.add_argument('--datadir', help="nuScenes dataset root directory",
                        default='/YOUR PATH', type=str)
    parser.add_argument('--outdir', help="Output directory for YOLO structure",
                        default='/YOUR PATH 1', type=str)
    return parser.parse_args()


def copy_images(data_dir, out_dir):

    # Loading datasets
    nusc_mini = NuScenes(version='v1.0-mini',     dataroot=data_dir, verbose=True)
    nusc      = NuScenes(version='v1.0-trainval', dataroot=data_dir, verbose=True)

    # Retrieving CAM_FRONT key frame tokens
    mini_tokens = set(s['token'] for s in nusc_mini.sample_data
                      if s['channel'] == 'CAM_FRONT' and s['is_key_frame'])
    all_tokens  =    [s['token'] for s in nusc.sample_data
                      if s['channel'] == 'CAM_FRONT' and s['is_key_frame']]

    # Splitting into train / val / test sets
    nusc_test        = NuScenes(version='v1.0-test', dataroot=data_dir, verbose=True)
    test_tokens      = [s['token'] for s in nusc_test.sample_data
                        if s['channel'] == 'CAM_FRONT' and s['is_key_frame']]
    train_tokens     = [t for t in all_tokens if t not in mini_tokens]
    val_tokens       = list(mini_tokens)

    sets      = ['train',        'val',       'test']
    all_sets  = [train_tokens,   val_tokens,  test_tokens]
    all_nuscs = {'train': nusc,  'val': nusc, 'test': nusc_test}

    for data_set, token_list in zip(sets, all_sets):
        nusc_cur = all_nuscs[data_set]

        # Create output directories
        cam_out_dir   = os.path.join(out_dir, 'images',       data_set)
        radar_out_dir = os.path.join(out_dir, 'images_radar', data_set)
        os.makedirs(cam_out_dir,   exist_ok=True)
        os.makedirs(radar_out_dir, exist_ok=True)

        print('Starting %s — %d samples' % (data_set, len(token_list)))
        num_copied  = 0
        num_missing = 0

        for cam_token in tqdm(token_list):
            try:
                sample_data = nusc_cur.get('sample_data', cam_token)
                sample      = nusc_cur.get('sample', sample_data['sample_token'])
                radar_token = sample['data']['RADAR_FRONT']
                pc_rec      = nusc_cur.get('sample_data', radar_token)
            except Exception as e:
                print("Token error %s : %s" % (cam_token, e))
                continue

            # Basename (consistent for camera, radar, and labels)
            img_basename   = os.path.basename(sample_data['filename'])         # xxx.jpg
            base_name      = img_basename.replace('.jpg', '')

            # Source paths
            cam_src   = os.path.join(data_dir, sample_data['filename'])
            radar_src = os.path.join(
                data_dir,
                pc_rec['filename'].replace('samples', 'imagepc_%02d' % RADIUS).replace('pcd', 'png')
            )

            # Destination paths
            cam_dst   = os.path.join(cam_out_dir,   img_basename)
            radar_dst = os.path.join(radar_out_dir, base_name + '.png')

            # Verify that label exists (if not, no need to copy images)
            label_path = os.path.join(out_dir, 'labels', data_set, base_name + '.txt')
            if not os.path.isfile(label_path):
                continue

            # Verify that source images exist
            if not os.path.isfile(cam_src):
                print("Missing camera image: %s" % cam_src)
                num_missing += 1
                continue
            if not os.path.isfile(radar_src):
                print("Missing radar image: %s" % radar_src)
                num_missing += 1
                continue

            # Copy files
            shutil.copy2(cam_src,   cam_dst)
            shutil.copy2(radar_src, radar_dst)
            num_copied += 1

        print("  Copied  : %d" % num_copied)
        print("  Missing : %d" % num_missing)

    print("Finished. Final structure:")
    print("  %s/images/train|val|test        -> camera images" % out_dir)
    print("  %s/images_radar/train|val|test  -> radar images r%02d" % (out_dir, RADIUS))
    print("  %s/labels/train|val|test        -> YOLO labels" % out_dir)


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "nuscenes":
        copy_images(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
