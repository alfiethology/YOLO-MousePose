# RatCam1 six-point adaptation

The fork now supports your existing rat annotations without relabelling the
landmarks. The keypoint order is:

1. `snout`
2. `left_ear`
3. `right_ear`
4. `neck_base`
5. `body_centre`
6. `tail_base`

Ear indices swap on horizontal flips; the other landmark identities stay fixed.
Skeleton edges follow `../skeleton_templates/rat.skeleton.json`. The original
seven-point mouse model YAMLs are retained; rat models are separate
`models/hub/YOLO-RatPose-{T,S,M,L}.yaml` files.

## Prepared data

`data/rat_kpts.yaml` points to the exported dataset at
`../models/datasets/pose/ratcam1_mousepose_rat6_v1/`:

| Split | Full frames | Rat skeletons |
|---|---:|---:|
| Training | 54 | 150 |
| Validation | 14 | 37 |

Source recordings stay entirely within one split. All keypoint coordinates,
visibility flags and upright enclosing boxes were exported from the existing
LabelMe skeleton shapes. The source annotations were not modified. Raw images
are full frames, with multiple animals supported per image.

The current 81 source frames contain 234 skeletons. Thirteen entire frames were
excluded because they contain at least one pose already marked
`exclude_from_pose_training`. This avoids keeping an image while removing the
label of a real animal in that image. The export records these decisions in
`excluded_frames.json`.

The source files still carry `pose_review_incomplete=true`. Therefore the export
is explicitly `review_required: true`: check that every visible animal has a
skeleton before using it as final training data. This adaptation has not declared
the full-frame pose review complete. Unlabelled keypoints (`v=0`) are masked;
labelled occluded points (`v=1`) and visible points (`v=2`) both supervise position.
The legacy head predicts whether a point is labelled, rather than a separate
visible-versus-occluded classification.

Re-export after review into a **new** output directory, using:

```bash
cd /home/or22503/OneDrive/vikki_rat_project/YOLO-MousePose
/home/or22503/miniconda3/envs/yolo-env/bin/python data/dataset/prepare_rat_dataset.py --help
```

The export refuses to overwrite existing datasets. Its manifest, source hashes
and validation report are stored alongside the dataset.

## Starting weights

The supplied `weights/yolobackbone.pth` is a classification/backbone checkpoint,
**not the paper's trained mouse pose model**. It only partially matches the rat
Tiny architecture: the current transfer loads 80 of 812 state tensors. All six
landmark outputs are newly initialized. This is not mouse-pose-pretrained fine-tuning.

`utils/pose_transfer.py` writes a transfer report for each run. It loads only
compatible tensors and never reuses a backbone checkpoint's optimizer, epoch
counter or learning-rate schedule for a fresh experiment. No weights download is
performed automatically. A genuine mouse-pose checkpoint can be supplied later,
but its architecture and keypoint semantics must be checked first.

The upstream head mixed box and landmark branch channels across anchors. This
adaptation corrects that grouping and marks new models `pose_anchor_layout:
per_anchor`. Consequently, old-format mouse pose heads are skipped conservatively;
their compatible feature layers can still transfer. Five shared landmark outputs
can be mapped from a checkpoint explicitly using the corrected layout. The new
neck-base output remains newly initialized.

## Training

Use the existing `yolo-env`; no replacement of its PyTorch/CUDA packages is needed.
The launcher resolves paths from the fork, so it works when invoked from elsewhere.

After source completeness review, export reviewed data and point `--data` at its
YAML. For a knowingly provisional experiment with the current candidate data:

```bash
/home/or22503/miniconda3/envs/yolo-env/bin/python \
  /home/or22503/OneDrive/vikki_rat_project/YOLO-MousePose/train_rat.py \
  --epochs 80 --allow-incomplete-review
```

The acknowledgement is needed because the source files explicitly retain their
pending-review flag. Without it, training stops with a clear message. No long
training has been started as part of adapting this fork.

Defaults are Tiny, AdamW, initial learning rate 0.0003, cosine decay, three warmup
epochs, batch 8, image size 640 and GPU 0. Use `--dry-run` to print the command,
`--cfg` to choose another rat model, `--weights` to choose an available matching
checkpoint, and `--data` to select a new export. See `data/hyp.rat.yaml` for the
complete initial recipe; it is a starting point, not a tuned result.

`best.pt` is selected using pose PCK at 0.1 of the ground-truth box diagonal,
including missed animals in its denominator. Direct `train.py --best-metric
bbox_map` instead selects box AP50:95. `metrics.csv` and `pose_metrics.json`
explicitly distinguish box detection metrics, pose PCK and localization error.
The uniform keypoint tolerance settings are configurable assumptions, not measured
rat anatomical uncertainty. These figures should not be equated directly with
the paper's reported scores.

Checkpoints are saved via temporary files and atomic replacement. Each fresh
experiment gets a new run directory. TensorBoard and W&B are optional, and the
rat launcher disables online W&B logging by default.

## Validation and inference

From the fork directory, after training:

```bash
/home/or22503/miniconda3/envs/yolo-env/bin/python test.py \
  --data data/rat_kpts.yaml --weights PATH/TO/best.pt --kpt-label --device 0

/home/or22503/miniconda3/envs/yolo-env/bin/python detect.py \
  --weights PATH/TO/best.pt --source PATH/TO/VIDEO.mp4 --kpt-label --device 0
```

Evaluation matches multiple animals by geometry and class before scoring their
landmarks. Unlabelled points do not count as errors; missed labelled animals
reduce PCK. Left/right ears and skeleton drawing use the six-point schema.

Focused regression tests:

```bash
cd /home/or22503/OneDrive/vikki_rat_project/YOLO-MousePose
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/or22503/miniconda3/envs/yolo-env/bin/python -m unittest discover -s tests -v
```

Integration evidence is in `../logs/ratpose_adapter_20261029/`. Short one-epoch,
128-pixel CPU runs are functionality checks, not useful trained pose checkpoints
or accuracy estimates.
