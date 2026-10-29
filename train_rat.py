#!/usr/bin/env python3
"""Launch the six-keypoint RatPose adaptation using explicit local paths."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/yolobackbone.pth')
    parser.add_argument('--cfg', type=Path, default=ROOT/'models/hub/YOLO-RatPose-T.yaml')
    parser.add_argument('--data', type=Path, default=ROOT/'data/rat_kpts.yaml')
    parser.add_argument('--name', default='rat6_backbone_finetune')
    parser.add_argument('--project', type=Path, default=ROOT/'runs/rat_pose')
    parser.add_argument('--allow-incomplete-review', action='store_true',
                        help='acknowledge exported full-frame pose annotations still require completeness review')
    parser.add_argument('--dry-run', action='store_true', help='print command only')
    args = parser.parse_args()
    command = [sys.executable, str(ROOT/'train.py'), '--cfg', str(args.cfg.resolve()),
               '--weights', str(args.weights.resolve()), '--data', str(args.data.resolve()),
               '--hyp', str(ROOT/'data/hyp.rat.yaml'), '--kpt-label', '--adamW',
               '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
               '--img-size', str(args.img_size), '--device', args.device,
               '--workers', str(args.workers), '--val-period', '1',
               '--project', str(args.project.resolve()), '--name', args.name]
    if args.allow_incomplete_review:
        command.append('--allow-incomplete-review')
    if args.dry_run:
        import shlex
        print(shlex.join(command))
        return 0
    environment = dict(os.environ)
    environment.setdefault('WANDB_MODE', 'disabled')
    environment.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    environment.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    Path(environment['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == '__main__':
    sys.exit(main())
