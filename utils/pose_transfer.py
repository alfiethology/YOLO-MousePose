"""Conservative transfer of compatible backbone and pose checkpoint weights.

The supplied yolobackbone.pth is an ImageNet backbone, not a mouse-pose model.
Legacy upstream pose checkpoints also use a different anchor-channel layout;
their detection/keypoint heads are deliberately not transferred automatically.
"""

from collections.abc import Mapping

import torch
from torch import nn


MOUSE_KEYPOINTS = ('snout', 'right_ear', 'right_hip', 'tail_base',
                   'left_hip', 'left_ear', 'body_centre')


def _name(value):
    return {'body_center': 'body_centre', 'centre_back': 'body_centre',
            'back_centre': 'body_centre'}.get(value, value)


def transfer_pose_weights(model, checkpoint, source_keypoints=None):
    """Load compatible tensors and return an auditable transfer report.

    ``checkpoint`` can be a torch checkpoint dict, a model, or a state dict.
    Shared landmark output channels can be remapped only for pose checkpoints
    explicitly marked ``pose_anchor_layout: per_anchor``. Unmatched landmarks,
    including a new neck_base output, retain their random initialization.
    """
    target = model.module if hasattr(model, 'module') else model
    source = checkpoint
    source_yaml = {}
    if isinstance(checkpoint, Mapping):
        for key in ('ema', 'model', 'state_dict'):
            if checkpoint.get(key) is not None:
                source = checkpoint[key]
                break
        source_yaml = checkpoint.get('model_yaml', checkpoint.get('yaml', {})) or {}
    if hasattr(source, 'module'):
        source = source.module
    if isinstance(source, nn.Module):
        source_yaml = getattr(source, 'yaml', source_yaml)
        source = source.float().state_dict()
    if not isinstance(source, Mapping):
        raise TypeError('Checkpoint must contain a model or state_dict mapping')
    source = {key.removeprefix('module.'): value for key, value in source.items()
              if isinstance(value, torch.Tensor)}
    if not source:
        raise ValueError('Checkpoint contains no model tensors')
    destination = target.state_dict()
    head = target.model[-1]
    prefix = f'model.{len(target.model)-1}.'
    has_pose_head = any(key.startswith(prefix+'m_kpt.') for key in source)
    layout = source_yaml.get('pose_anchor_layout')
    if isinstance(checkpoint, Mapping):
        layout = checkpoint.get('pose_anchor_layout', layout)
    safe_pose_layout = layout == 'per_anchor'
    source_names = source_keypoints or source_yaml.get('kpt_names')
    if source_names and isinstance(source_names, dict):
        source_names = source_names.get(0, source_names.get('0'))
    target_names = target.yaml.get('kpt_names')
    if source_names:
        source_names = [_name(name) for name in source_names]
    if target_names:
        target_names = [_name(name) for name in target_names]
    shared = {i: source_names.index(name) for i, name in enumerate(target_names or [])
              if source_names and name in source_names}
    loaded, skipped, remapped = {}, [], []
    for key, value in source.items():
        if key not in destination or 'anchor' in key:
            continue
        is_head = key.startswith(prefix)
        if is_head and has_pose_head and not safe_pose_layout:
            skipped.append(key)
            continue
        if key.startswith(prefix+'m_kpt.') and has_pose_head:
            # Only the final Conv2d produces per-keypoint output channels;
            # intermediate keypoint-branch layers transfer by shape normally.
            suffix = key[len(prefix):].rsplit('.', 1)[0]
            module = head.get_submodule(suffix)
            final_convs = [branch[-1] if isinstance(branch, nn.Sequential) else branch
                           for branch in head.m_kpt]
            if module in final_convs:
                if not source_names or not target_names or not shared:
                    skipped.append(key)
                    continue
                expected_source = head.na * len(source_names) * 3
                if value.shape[0] != expected_source or value.shape[1:] != destination[key].shape[1:]:
                    skipped.append(key)
                    continue
                adapted = destination[key].clone()
                for anchor in range(head.na):
                    for new_index, old_index in shared.items():
                        to = (anchor*head.nkpt+new_index)*3
                        start = (anchor*len(source_names)+old_index)*3
                        adapted[to:to+3] = value[start:start+3].to(adapted)
                loaded[key] = adapted
                remapped.append(key)
                continue
        if value.shape == destination[key].shape:
            loaded[key] = value.to(destination[key])
        else:
            skipped.append(key)
    if not loaded:
        raise ValueError('No compatible checkpoint tensors found for this architecture')
    target.load_state_dict(loaded, strict=False)
    report = dict(loaded_tensors=len(loaded), target_tensors=len(destination),
                  loaded_parameters=sum(value.numel() for value in loaded.values()),
                  remapped_head_tensors=remapped, skipped_tensors=skipped,
                  source_has_pose_head=has_pose_head,
                  source_pose_anchor_layout=layout,
                  legacy_pose_head_skipped=bool(has_pose_head and not safe_pose_layout),
                  copied_landmarks=[target_names[i] for i in sorted(shared)] if remapped else [],
                  new_landmarks=[name for i, name in enumerate(target_names or []) if i not in shared]
                      if remapped else list(target_names or []))
    return report
