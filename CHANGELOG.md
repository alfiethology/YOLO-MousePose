# Changelog

Changes to this fork are recorded below. The rat adaptation is a local,
unreleased change to the original YOLO-MousePose project.

## Unreleased — 2026-10-29

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
