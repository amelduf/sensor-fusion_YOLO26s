"""
filter_dataset_size.py: Filter out tiny bounding boxes and empty annotated images
=================================================================================
- Removes bounding boxes (bboxes) where w < min_size OR h < min_size.
- Deletes triplets (camera image + radar image + label) if no bboxes remain.

Usage:
  # Run a Dry Run first
  python3 filter_dataset_size.py \
      --datadir /YOUR PATH \
      --splits test --min_size 0.05 --dry_run

  # Apply actual filtering
  python3 filter_dataset_size.py \
      --datadir /YOUR PATH \
      --splits test --min_size 0.05

  # Multiple datasets
  python3 filter_dataset_size.py \
      --datadir \
          /YOUR PATH1 \
          /YOUR PATH2 \
          /YOUR PATH3 \
          /YOUR PATH4 \
      --splits test --min_size 0.05
"""

import os
import argparse
from tqdm import tqdm


def filter_labels(label_path, min_size):
    """
    Filters out bboxes that are too small in a label file.
    Returns (new_lines, total_count, kept_count, filtered_count).
    """
    if not os.path.isfile(label_path):
        return [], 0, 0, 0

    with open(label_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        return [], 0, 0, 0

    new_lines  = []
    n_total    = 0
    n_kept     = 0
    n_filtered = 0

    for line in lines:
        parts = line.split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue

        n_total += 1

        if len(vals) == 5:
            # Detection format: cls cx cy w h
            cls, cx, cy, w, h = vals
            if w >= min_size and h >= min_size:
                new_lines.append(line)
                n_kept += 1
            else:
                n_filtered += 1

        elif len(vals) > 5 and (len(vals) - 1) % 2 == 0:
            # Segmentation format → calculate bounding box
            coords = vals[1:]
            xs = coords[0::2]
            ys = coords[1::2]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w >= min_size and h >= min_size:
                new_lines.append(line)
                n_kept += 1
            else:
                n_filtered += 1

    return new_lines, n_total, n_kept, n_filtered


def filter_split(datadir, split, min_size, dry_run=False):
    """Filters a specific split of a dataset."""
    img_dir   = os.path.join(datadir, 'images',       split)
    radar_dir = os.path.join(datadir, 'images_radar', split)
    label_dir = os.path.join(datadir, 'labels',       split)

    if not os.path.isdir(label_dir):
        print("  Label directory not found: %s" % label_dir)
        return

    files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    if not files:
        print("  No .txt files found.")
        return

    n_images_total    = len(files)
    n_images_kept     = 0
    n_images_deleted  = 0
    n_bbox_total      = 0
    n_bbox_kept       = 0
    n_bbox_filtered   = 0

    for fname in tqdm(files, desc='  %s/%s' % (os.path.basename(datadir), split)):
        base       = os.path.splitext(fname)[0]
        label_path = os.path.join(label_dir, fname)
        cam_path   = os.path.join(img_dir,   base + '.png')
        if not os.path.exists(cam_path):
            cam_path = os.path.join(img_dir, base + '.jpg')
        radar_path = os.path.join(radar_dir, base + '.png')

        new_lines, n_tot, n_kept, n_filt = filter_labels(label_path, min_size)

        n_bbox_total    += n_tot
        n_bbox_kept     += n_kept
        n_bbox_filtered += n_filt

        if len(new_lines) == 0:
            # No bboxes left → delete the triplet (label, camera, radar)
            n_images_deleted += 1
            if not dry_run:
                for path in [label_path, cam_path, radar_path]:
                    if os.path.isfile(path):
                        os.remove(path)
        else:
            # Update the label file
            n_images_kept += 1
            if not dry_run and n_filt > 0:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(new_lines) + '\n')

    print("\n  ── Summary for %s/%s ──" % (os.path.basename(datadir), split))
    print("  Total images        : %d" % n_images_total)
    print("  Images kept         : %d" % n_images_kept)
    print("  Images deleted      : %d (no bboxes remaining)" % n_images_deleted)
    print("  Total Bboxes        : %d" % n_bbox_total)
    print("  Bboxes kept         : %d (%.1f%%)" %
          (n_kept, 100 * n_bbox_kept / max(n_bbox_total, 1)))
    print("  Bboxes filtered     : %d (too small)" % n_bbox_filtered)
    if dry_run:
        print("  (DRY RUN — no files were modified)")


def filter_dataset(datadir, splits, min_size, dry_run=False):
    print("\n" + "═" * 55)
    print("  Dataset  : %s" % datadir)
    print("  Min size : %.1f%% (%.0fpx for 640 input)" %
          (min_size * 100, min_size * 640))
    print("  Mode     : %s" % ('DRY RUN' if dry_run else 'ACTUAL FILTERING'))
    print("═" * 55)

    for split in splits:
        split_path = os.path.join(datadir, 'labels', split)
        if os.path.isdir(split_path):
            filter_split(datadir, split, min_size, dry_run)
        else:
            print("  Split '%s' not found, skipping..." % split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir',   nargs='+', required=True)
    parser.add_argument('--splits',    nargs='+', default=['test'])
    parser.add_argument('--min_size',  type=float, default=0.05,
                        help='Min bbox size as fraction (0.05 = 5% = 32px on 640px image)')
    parser.add_argument('--dry_run',   action='store_true')
    args = parser.parse_args()

    for datadir in args.datadir:
        filter_dataset(datadir, args.splits, args.min_size, args.dry_run)

    print("\n" + "═" * 55)
    print("Done!")
