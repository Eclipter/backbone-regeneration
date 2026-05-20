"""Public inference package for backbone regeneration."""

from ._pynamod_bootstrap import ensure_pynamod_importable

ensure_pynamod_importable()

from .inference import (
    OnnxSampler,
    predict_backbone,
    predict_backbone_trajectory,
    write_structure,
    write_trajectory_frames_directory,
)

__all__ = [
    'OnnxSampler',
    'predict_backbone',
    'predict_backbone_trajectory',
    'write_structure',
    'write_trajectory_frames_directory',
]
