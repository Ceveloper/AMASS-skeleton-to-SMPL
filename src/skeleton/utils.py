import numpy as np


def get_limb_length(
    skeletons: np.ndarray, limbseq: list[tuple[int, int]]
) -> np.ndarray:
    """
    Get mean limb length.
    Args:
        skeletons: Skeletons.
        limbseq: Limb sequence.
    Returns:
        limb_length: Limb length.
    """
    limb_lengths = []
    for start_idx, end_idx in limbseq:
        limb_lengths.append(
            np.linalg.norm(skeletons[:, start_idx] - skeletons[:, end_idx], axis=-1)
        )
    return np.stack(limb_lengths, axis=-1)
