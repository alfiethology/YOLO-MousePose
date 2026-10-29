#!/usr/bin/env python3
"""Convert the rat LabelMe skeletons to six-point YOLO-MousePose labels.

Full camera frames are preserved. Source recordings are grouped before splitting.
This prepares a candidate dataset; it does not certify full-frame completeness.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
KEYPOINTS = ['snout', 'left_ear', 'right_ear', 'neck_base', 'body_centre', 'tail_base']
FLIP_IDX = [0, 2, 1, 3, 4, 5]
EDGES = [[4, 3], [3, 1], [1, 0], [2, 0], [2, 3], [5, 4]]
VIDEO = re.compile(r'RatCam\d+_cam\d+_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def yolo_row(shape, width, height):
    pose = shape['pose']
    if pose['keypoints'] != KEYPOINTS or len(shape['points']) != 4 + len(KEYPOINTS):
        raise ValueError('Skeleton keypoint order does not match the rat schema')
    if pose.get('flip_idx', FLIP_IDX) != FLIP_IDX:
        raise ValueError('Unexpected left/right keypoint mapping')
    visibility = pose['visibility']
    if len(visibility) != len(KEYPOINTS) or any(v not in (0, 1, 2) for v in visibility):
        raise ValueError('Expected six visibility flags with values 0, 1 or 2')
    xs, ys = zip(*shape['points'][:4])
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Invalid pose bounding box')
    values = [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
              (x2 - x1) / width, (y2 - y1) / height]
    for (x, y), visible in zip(shape['points'][4:], visibility):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError('Nonfinite keypoint')
        if visible:
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError('Labelled keypoint outside frame')
            values.extend([x / width, y / height, visible])
        else:
            # Do not invent coordinates for an unlabelled landmark.
            values.extend([0, 0, 0])
    return '0 ' + ' '.join(f'{value:.10f}' for value in values)


def assignments(records, seed, fraction):
    groups = defaultdict(list)
    for record in records:
        groups[record['recording']].append(record)
    keys = sorted(groups)
    rng = random.Random(seed)
    total_frames = len(records)
    total_animals = sum(len(record['shapes']) for record in records)
    best = None
    for _ in range(5000):
        validation = {key for key in keys if rng.random() < fraction}
        count = sum(len(groups[key]) for key in validation)
        animals = sum(len(record['shapes']) for key in validation for record in groups[key])
        if count in (0, total_frames):
            continue
        score = (count / total_frames - fraction) ** 2 + (animals / total_animals - fraction) ** 2
        if best is None or score < best[0]:
            best = (score, validation)
    if best is None:
        raise ValueError('Cannot create separate recording-grouped train/validation sets')
    return {key: 'val' if key in best[1] else 'train' for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT / 'frames/ratcam1/labelled/pose_estimation')
    parser.add_argument('--output', type=Path, default=PROJECT / 'models/datasets/pose/ratcam1_mousepose_rat6_v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--config', type=Path, default=REPO / 'data/rat_kpts.yaml')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite dataset: {output}')
    if not 0 < args.val_fraction < 1:
        parser.error('Validation fraction must lie between zero and one')
    records, excluded, hashes, seen = [], [], {}, set()
    original_counts = Counter()
    for annotation in sorted(source.rglob('*.json')):
        document = json.loads(annotation.read_text())
        shapes = [shape for shape in document.get('shapes', []) if shape.get('shape_type') == 'skeleton']
        if not shapes:
            continue
        image = annotation.with_suffix('.jpg')
        match = VIDEO.search(image.stem)
        if not match:
            raise ValueError(f'Unknown source recording: {image}')
        hashes[str(annotation)] = sha(annotation)
        hashes[str(image)] = sha(image)
        original_counts['frames'] += 1
        original_counts['animals'] += len(shapes)
        if any(shape.get('flags', {}).get('exclude_from_pose_training', False) for shape in shapes):
            excluded.append(dict(annotation=str(annotation), image=str(image), recording=match.group(),
                                 skeletons=len(shapes), reason='Frame contains a pose explicitly excluded from training; omit the entire frame rather than treat that visible animal as background'))
            continue
        with Image.open(image) as im:
            width, height = im.size
            im.verify()
        if (width, height) != (document['imageWidth'], document['imageHeight']):
            raise ValueError(f'Annotation/image dimensions disagree: {image}')
        if hashes[str(image)] in seen:
            raise ValueError(f'Duplicate source image: {image}')
        seen.add(hashes[str(image)])
        rows = [yolo_row(shape, width, height) for shape in shapes]
        records.append(dict(image=image, annotation=annotation, rows=rows, shapes=shapes,
                            recording=match.group(), lighting=annotation.parent.name,
                            review_required=bool(document.get('flags', {}).get('pose_review_incomplete'))))
    if not records:
        raise ValueError('No eligible pose frames found')
    splits = assignments(records, args.seed, args.val_fraction)
    for split in ['train', 'val']:
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
    manifest, counts, visibility = [], defaultdict(Counter), Counter()
    for record in records:
        split = splits[record['recording']]
        image = record['image']
        shutil.copy2(image, output / 'images' / split / image.name)
        (output / 'labels' / split / (image.stem + '.txt')).write_text('\n'.join(record['rows']) + '\n')
        counts[split]['frames'] += 1
        counts[split]['animals'] += len(record['shapes'])
        counts[split][record['lighting'] + '_frames'] += 1
        for shape in record['shapes']:
            visibility.update(shape['pose']['visibility'])
        manifest.append(dict(image=f'images/{split}/{image.name}', split=split, recording=record['recording'],
                             annotation=str(record['annotation']), original_image=str(image), sha256=sha(image),
                             skeletons=len(record['shapes']), review_required=record['review_required']))
    with (output / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    review_required = any(record['review_required'] for record in records)
    config = (
        '# Candidate rat pose data. Source completeness review is still required.\n'
        f'train: {json.dumps(str(output / "images/train"))}\n'
        f'val: {json.dumps(str(output / "images/val"))}\n'
        'nc: 1\nnames: [rat]\nnkpt: 6\nkpt_shape: [6, 3]\n'
        f'keypoint_names: {json.dumps(KEYPOINTS)}\n'
        f'kpt_names: {json.dumps(KEYPOINTS)}\n'
        f'flip_idx: {json.dumps(FLIP_IDX)}\nskeleton: {json.dumps(EDGES)}\n'
        # Uniform tolerance is a configuration choice, not measured rat variability.
        'kpt_oks_sigmas: [0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667]\n'
        f'review_required: {str(review_required).lower()}\n'
        f'preparation_manifest: {json.dumps(str(output / "manifest.csv"))}\n'
    )
    (output / 'data.yaml').write_text(config)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(config)
    summary = dict(status='candidate dataset; pending source completeness review' if review_required else 'exported',
                   source_counts=dict(original_counts), counts={key:dict(value) for key,value in counts.items()},
                   excluded_whole_frames=len(excluded), visibility_counts=dict(visibility),
                   recordings={split:sum(value == split for value in splits.values()) for split in ['train','val']},
                   review_required=review_required, keypoints=KEYPOINTS, seed=args.seed,
                   source_unchanged=True)
    write_json(output / 'summary.json', summary)
    write_json(output / 'excluded_frames.json', excluded)
    write_json(output / 'source_hashes.json', hashes)
    for path, expected in hashes.items():
        assert sha(Path(path)) == expected, f'Source file changed: {path}'
    (output / 'README.md').write_text(
        '# RatCam1 six-point YOLO-MousePose candidate dataset\n\n'
        'Full frames copied without cropping or resizing. Each rat label contains an upright '
        'box and six keypoints in this order: snout, left_ear, right_ear, neck_base, body_centre, '
        'tail_base. Visibility 0 is unlabelled, 1 labelled but occluded, and 2 visible. The '
        'loader trains coordinates for v=1/2 and masks v=0. The legacy model has a labelled '
        'keypoint confidence, not a separate visible-versus-occluded classifier.\n\n'
        'Source recording groups stay together across train and validation. All sources '
        'currently carry pose_review_incomplete=true; additional visible rats may need '
        'skeletons. This export therefore remains review_required=true. It is not a claim '
        'that all visible rats in the source frames have been annotated.\n\n'
        'Frames containing any pose with exclude_from_pose_training=true are omitted '
        'entirely, preserving the existing perspex exclusions without treating the omitted '
        'animal as background. See excluded_frames.json. Source annotations are unchanged.\n\n'
        'Uniform OKS sigmas are explicit configurable tolerances, not rat-specific empirical '
        'measurements. Compare methods using identical sigmas and the same held-out data.\n'
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
