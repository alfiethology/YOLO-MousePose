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

## Current default: four-rat full frames, v3

`train_rat.py` uses `data/rat_fullframes_kpts.yaml`: the **exact same image and
TXT label files** as the latest YOLO26m full-frame experiment, at
`../models/datasets/pose/ratcam1_rat6_four_rats_fullframes_v3/`.
There are **40 training frames (160 rats)** and **18 validation frames (72 rats)**.
Recordings remain separated using the existing split. Nine frames with explicit
pose-exclusion flags were already omitted by the export. The original annotations
and all previous datasets are unchanged. No cropping or resplitting is performed.

From the project root:

```bash
conda activate yolo-env
python YOLO-MousePose/train_rat.py --epochs 150 --name rat6_fullframes_v3_150e
```

Defaults: 928-pixel input, batch 8, one update per minibatch, AdamW LR 0.001,
cosine decay, augmentation, six rat points, existing Tiny model/backbone weights.
The fork requires a positive batch size; Ultralytics automatic batch `-1` is not
supported here. Results go to `runs/rat_pose/rat6_fullframes_v3_150e/` inside this
fork. Use `--dry-run` to inspect the command. To repeat the previous crop experiment,
pass `--data YOLO-MousePose/data/rat_crops_kpts.yaml --img-size 640` explicitly.

MousePose uses separate `<split>.mousepose.cache` label caches so its Torch cache
format cannot collide with Ultralytics' `<split>.cache` files in the shared export.
The dataset metadata uses a list of six keypoint names for MousePose's metric CSV.
Its PCK and the YOLO26 pose OKS mAP are different metrics; the two models also use
different losses, capacity and starting weights, so compare predictions as well.

## Previous crop experiment: pose crops retaining neighbours

The previous crop configuration `data/rat_crops_kpts.yaml` points to
`../models/datasets/pose/ratcam1_mousepose_rat6_crowded_crops_v3/`.
The export uses 129 full frames / 357 skeletons: **186 training crops and 61
validation crops**. Only 17 explicitly excluded poses are omitted as focal crops.
Nearby rats are retained, including 82 crops with known neighbours. Crops expand
to include whole known neighbours; 93 identical group crops are deduplicated.

Every saved skeleton intersecting a crop contributes its whole bounding box and
available in-crop landmarks. Box-only neighbours contribute detection targets
with all pose landmarks unavailable. Explicitly excluded poses can contribute
boxes, but their pose points are not supervised. Unknown animals lacking any
annotation can remain unlabelled; inspect the previews. The preview overlays show
focal landmarks, while the TXT files also include known neighbours.

Source recordings stay together using the shared rat/ID split and deterministic
fallback assignments. Sources are unchanged. Regenerate to a new output directory
with `data/dataset/prepare_rat_crowded_crops.py --output <new-directory>`.
Open `preview_overview.jpg` and `previews/` in the dataset folder.
The earlier isolated `ratcam1_mousepose_rat6_crops_v1` export is preserved.

## Earlier full-frame candidate export

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

For a shorter 30-epoch trial using the default four-rat full-frame dataset:

```bash
/home/or22503/miniconda3/envs/yolo-env/bin/python \
  /home/or22503/OneDrive/vikki_rat_project/YOLO-MousePose/train_rat.py \
  --epochs 30 --name rat6_fullframes_v3_30e
```

If you explicitly choose the earlier full-frame candidate YAML with `--data`,
`--allow-incomplete-review` is still needed because those source files retain
their pending completeness-review flag. No long training was started during
crop preparation.

Defaults are Tiny, AdamW, initial learning rate 0.001, cosine decay, three warmup
epochs, batch 8, image size 928 and GPU 0. Use `--dry-run` to print the command,
`--cfg` to choose another rat model, `--weights` to choose an available matching
checkpoint, and `--data` to select a new export. See `data/hyp.rat.yaml` for the
complete initial recipe; it is a starting point, not a tuned result.

Training now updates on every minibatch by default (`--accumulate 1`). With only
11 minibatches per epoch, the inherited nominal-batch-64 accumulation performed
very few updates and discarded some unfinished accumulated gradients at epoch
boundaries. `metrics.csv` records the optimizer-step count. The affine filter
also retains the clipped box of a visible rat instead of deleting its entire
target when a tight crop is zoomed beyond the canvas. Individual points outside
the canvas are masked, so some partial skeletons in training previews are normal;
fully visible animals with no target are not.

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
