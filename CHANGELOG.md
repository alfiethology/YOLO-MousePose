# Changelog

Changes to this fork are recorded below. The rat adaptation is a local,
unreleased change to the original YOLO-MousePose project.

## Unreleased — 2026-10-29

### Training failure diagnosis and fixes

- Diagnosed visible rats losing their entire pose target during affine
  augmentation: the inherited candidate filter required over 80% of the
  transformed box to remain in the image. In a reproducible 352-sample replay,
  88 augmented crops lost their target despite a large rat remaining visible.
- Retained clipped pose boxes for visible animals and masked only landmarks
  outside the image. Fully out-of-frame or negligible boxes are still removed.
  The same replay produced zero target losses after the fix.
- Added three regression tests for partially clipped animals, fully outside
  animals, and zoomed pose crops retaining boxes while masking outside points.
- Made gradient accumulation explicit, defaulting to one optimizer update per
  minibatch for the small crop dataset. Remaining gradients are applied at the
  end of each epoch instead of being discarded. Metrics now record optimizer
  update counts.
- Revised the initial crop-training recipe to LR 0.001, 10-degree rotations,
  translation 0.05 and scale variation 0.15. The available checkpoint only
  partially initializes the network; the earlier fine-tuning recipe learned
  too slowly in the presence of corrupted augmented targets.
- Preserved the existing 10/100-epoch runs and saved diagnostic evidence under
  `../logs/ratpose_training_debug_20261029/`.

### Neighbour-screened crop dataset

- Added `data/dataset/prepare_rat_crops.py` to generate one pose-labelled animal
  per crop, with 10-pixel padding and translated six-point coordinates/visibility.
- Screened using reviewed master BBs on matching frames, the verified generic
  BB rat detector elsewhere, and all other pose boxes as additional evidence.
- Held out crops with neighbouring boxes entering the crop, unmatched focal
  boxes, or existing per-pose exclusions. Exclusions apply per animal rather than
  discarding unrelated isolated animals from the same frame.
- Exported **88 training and 21 validation crops**, retaining source-recording
  split assignments from the rat/ID datasets where available.
- Added an overview image, complete accepted/excluded preview sheets, a local
  preview index and detailed screening/provenance manifests.
- Updated the training defaults to `data/rat_crops_kpts.yaml`. The earlier
  incomplete full-frame export remains available explicitly.
- Added regression checks for crop intersections, focal-box matching, labelled
  landmarks beyond the BB and missing-point masking. Source annotations remain
  unchanged; automatic screening is not a claim of perfect neighbour detection.

### Rat skeleton and model configuration

- Added six-point rat configurations for Tiny, Small, Medium and Large models:
  `models/hub/YOLO-RatPose-{T,S,M,L}.yaml`.
- Defined the landmark order as `snout`, `left_ear`, `right_ear`, `neck_base`,
  `body_centre`, `tail_base`.
- Added the rat skeleton connections and left/right ear flip mapping.
- Retained the original seven-point mouse configurations.
- Related source-data update: renamed `centre_back` to `body_centre` in the
  81 pose annotation files and `skeleton_templates/rat.skeleton.json` outside
  this repository. Point coordinates, ordering and connections were unchanged.

### Model and loss corrections

- Replaced hard-coded seven-point assumptions with the configured keypoint count.
- Corrected detection/keypoint channel grouping so each anchor receives its own
  box and landmark outputs.
- Applied sigmoid to landmark confidence outputs during inference.
- Updated augmented inference to transform landmarks alongside bounding boxes.
- Masked unlabelled points in the coordinate loss while retaining supervision
  for labelled occluded points.
- Corrected the landmark-confidence loss and handling of points on image edges.
- Replaced blocking nonfinite-loss debugging with explicit errors.
- Made per-landmark loss tolerances configurable. The rat defaults are uniform
  assumptions, not measured anatomical uncertainty.

### Data loading and preparation

- Added `data/dataset/prepare_rat_dataset.py` and `data/rat_kpts.yaml` to export
  the existing LabelMe skeleton annotations as full-frame YOLO pose data.
- Added configurable keypoint counts, schema-aware caches, visibility parsing,
  six-point affine transformations and anatomically consistent ear flipping.
- Removed unused image-augmentation dependencies from the loader and updated
  deprecated NumPy/PyTorch usage.
- Kept source recordings together when creating training and validation splits.
- Prepared a candidate export with **54 training frames / 150 skeletons** and
  **14 validation frames / 37 skeletons**.
- Omitted **13 entire frames** containing explicitly excluded poses, avoiding
  removal of an animal's annotation while keeping that animal in a training image.
- Preserved source completeness-review flags. The candidate data remain marked
  `review_required: true`; no source pose annotations were changed by the export.
- Recorded provenance, source hashes, exclusions and validation results alongside
  the generated dataset outside this repository.

### Starting-weight handling

- Added `utils/pose_transfer.py` for compatible tensor transfer with a saved
  transfer report for each training run.
- Prevented fresh fine-tuning runs from inheriting a backbone checkpoint's
  optimizer, learning-rate schedule or epoch counter.
- Added shared-landmark output mapping for checkpoints explicitly using the
  corrected per-anchor output layout. Newly introduced landmarks retain their
  initial parameters.
- Conservatively skipped legacy pose heads with incompatible anchor-channel
  semantics instead of treating their outputs as directly compatible.
- Verified that the supplied `weights/yolobackbone.pth` is a backbone checkpoint,
  not the paper's trained mouse-pose checkpoint. It transfers **80 of 812 state
  tensors** into the Tiny rat configuration and supplies no pretrained landmark
  outputs.

### Training and checkpointing

- Added `train_rat.py`, `data/hyp.rat.yaml` and `requirements-rat.txt` for a
  reproducible local rat-training setup using the existing Python environment.
- Added explicit acknowledgement of pending source completeness review before
  training on the candidate export.
- Configured the launcher with Tiny, AdamW, initial learning rate 0.0003,
  cosine decay, three warmup epochs, batch size 8 and 640-pixel images.
- Made validation frequency configurable and ensured validation runs on the
  final epoch.
- Added named `metrics.csv` logging and selection of `best.pt` by pose PCK at
  0.1 of the ground-truth box diagonal; box AP50:95 remains an alternative.
- Corrected the previous fitness-metric indexing error.
- Added atomic checkpoint saving and optimizer stripping to avoid publishing
  partially written model files.
- Made TensorBoard optional and disabled online W&B logging by default in the
  rat launcher. Evaluation and inference no longer attempt dependency updates.

### Evaluation, inference and plots

- Added one-to-one geometric matching between predicted and labelled animals
  before scoring landmarks in multi-animal frames.
- Counted missed labelled animals in the PCK denominator and excluded unlabelled
  points from pose-error calculations.
- Reported box precision/recall/AP separately from pose PCK and matched-point
  localization error in `pose_metrics.json`.
- Updated NMS to retain each animal's complete keypoint triplets and correct
  detection confidence.
- Fixed detection/keypoint pairing when iterating over predictions.
- Updated skeleton drawing, pose exports and training plots for the six-point
  schema and named metric columns.
- Added explicit trusted-checkpoint loading for current PyTorch versions.

### Verification and documentation

- Added **10 passing regression tests** covering model outputs, loss masking,
  weight transfer, multi-animal matching, missed-animal penalties, pose exports
  and plotting.
- Completed short one-epoch CPU training/validation checks on the actual candidate
  dataset, followed by standalone validation and inference using a saved checkpoint.
- Verified that source pose images and annotation JSON were unchanged by adaptation
  and export, and that `git diff --check` passed.
- Added [RAT_SETUP.md](RAT_SETUP.md) with data preparation, training, validation,
  inference commands and weight-transfer limitations.
- Saved integration evidence in `../logs/ratpose_adapter_20261029/`.

### Remaining limitations

- A genuine mouse-pose-trained checkpoint has not been obtained or verified.
  The supplied backbone provides only partial initialization.
- Full-frame pose completeness review remains pending according to the source
  annotation flags.
- The short CPU runs demonstrate functionality, not useful pose accuracy.
  No long rat-pose training run was started as part of this adaptation.
- The changed output layout and evaluation definitions mean results must not be
  equated directly with the original paper's reported scores.

## 2026-10-29 — retain neighbouring rats in pose crops

- Add `prepare_rat_crowded_crops.py`: retain close rats and export all known intersecting skeletons or box-only targets, masking unavailable and out-of-crop landmarks.
- Regenerate from 129 frames / 357 skeletons: 247 train and 93 validation focal crops; respect 17 existing exclusions.
- Keep recording groups separate and reuse shared rat/ID assignments. Switch the default crop YAML to the new export; preserve source annotations and the isolated export.

## 2026-10-29 — correct neighbour fragments in crop export

- Audit confirms all 129 sources are full 1296x972 frames and all 340 v2 crops reconstruct exactly from those sources; no double cropping.
- Expand crops to include complete known neighbouring boxes and labelled points, iterating through connected neighbours. The v2 export could supervise tiny neighbour fragments (106 of 211 neighbour occurrences contained less than 20% of the source box).
- Export v3 with 186 train / 61 validation crops after removing 93 duplicate group crops; keep all 17 explicit pose exclusions. Point the default YAML at v3 and preserve v2 and its training run.

## 2026-10-29 — four-rat full-frame v3 comparison

- Add `data/rat_fullframes_kpts.yaml` pointing directly to the same 40 train / 18 validation full-frame image and TXT files as YOLO26m. Adapt keypoint-name metadata to the fork.
- Default `train_rat.py` to full-frame v3, 928 pixels, 150 epochs and a distinct run name; retain batch 8, augmentation, cosine decay and one optimizer update per batch. Reject unsupported automatic batch sizes clearly.
- Give MousePose `.mousepose.cache` files to avoid incompatible Torch/NumPy label cache collisions with Ultralytics on shared data. Preserve Ultralytics caches and all source images/labels.
