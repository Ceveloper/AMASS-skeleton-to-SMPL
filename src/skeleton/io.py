from pathlib import Path

import numpy as np
import torch


def load_skeleton(
    npy_path: Path,
    unit: str = "m",
) -> np.ndarray:
    """
    Load skeleton from file.
    Args:
        npy_path: Path to the npy file.
        skeleton: Skeleton type.
        unit: Unit of the skeleton.
    Returns:
        joints: Joints of the skeleton.
    """
    # load skeleton from file
    if unit == "mm":
        scale_factor = 1000.0
    elif unit == "cm":
        scale_factor = 100.0
    elif unit == "m":
        scale_factor = 1.0
    else:
        raise ValueError(f"Unknown unit: {unit}")

    return np.load(npy_path, allow_pickle=True) / scale_factor


def remove_hip_motion(motion: torch.Tensor) -> torch.Tensor:
    """
    Remove hip translation from the skeletons.
    Args:
        skeletons: Skeletons (T, J, 3).
    Returns:
        skeletons: Skeletons without hip translation.
    """
    motion = motion - motion[..., 0:1, :]
    return motion
