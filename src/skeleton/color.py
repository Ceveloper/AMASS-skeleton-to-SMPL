import numpy as np
from omegaconf import DictConfig


def limb_colors(
    limb_positions: list[str],
    base: DictConfig,
    left: DictConfig | None = None,
    right: DictConfig | None = None,
) -> np.ndarray:
    """
    Get colors for the skeleton.
    Args:
        skeleton_type: Skeleton type.
        base_color: Base color of the skeleton.
        left_color: Left color of the skeleton.
        right_color: Right color of the skeleton.
    Returns:
        colors: Colors of the skeleton.
    """
    left = base if left is None else left
    right = base if right is None else right

    colors = []
    for limb_pos in limb_positions:
        if limb_pos == "l":
            colors.append(
                np.array(
                    [*left["rgb"], *left["hsv"], left["brightness"], left["contrast"]]
                )
            )
        elif limb_pos == "r":
            colors.append(
                np.array(
                    [
                        *right["rgb"],
                        *right["hsv"],
                        right["brightness"],
                        right["contrast"],
                    ]
                )
            )
        else:
            colors.append(
                np.array(
                    [*base["rgb"], *base["hsv"], base["brightness"], base["contrast"]]
                )
            )

    return np.array(colors)


def _metric_colors(
    metric: np.ndarray,
    base_color: np.ndarray,
    pos_color: np.ndarray,
    neg_color: np.ndarray,
    hsv: np.ndarray,
    brightness: np.ndarray,
    contrast: np.ndarray,
) -> np.ndarray:
    """
    Assign colors to the metric.
    Args:
        metric: Metric.
        base_color: Base color.
        pos_color: Positive color.
        neg_color: Negative color.
        hsv: HSV.
        brightness: Brightness.
        contrast: Contrast.
    Returns:
        colors: Colors.
    """
    colors = np.zeros_like(metric)[:, :, None].repeat(3, -1)  # (T, J, 3)
    T, J = metric.shape
    colors[metric > 0] = pos_color[None, None, :] * metric[
        metric > 0, None
    ] + base_color[None, None, :] * (1 - metric[metric > 0, None])
    colors[metric < 0] = neg_color[None, None, :] * (
        -metric[metric < 0, None]
    ) + base_color[None, None, :] * (1 + metric[metric < 0, None])

    # concatenate hsv, brightness, contrast
    colors = np.concatenate(
        [
            colors,
            np.concatenate([hsv, brightness[None], contrast[None]], axis=-1)[None, None]
            .repeat(T, 0)
            .repeat(J, 1),
        ],
        axis=-1,
    )

    return colors


def stretching_color(
    limb_lengths: np.ndarray,
    mean_limb_length: np.ndarray,
    max_strech: float,
    base_color: np.ndarray,
    pos_color: np.ndarray,
    neg_color: np.ndarray | None = None,
    hsv: np.ndarray = np.array([1.0, 1.0, 1.0]),
    brightness: np.ndarray = np.array([1.0]),
    contrast: np.ndarray = np.array([1.0]),
) -> np.ndarray:
    """
    Get colors for the skeleton.
    Args:
        skeletons: Skeletons.
        mean_skeletion: Mean skeleton.
        base_color: Base color of the skeleton.
        pos_color: Positive color of the skeleton.
        neg_color: Negative color of the skeleton.
    Returns:
        colors: Colors of the skeleton.
    """
    stretching = (limb_lengths - mean_limb_length[None]) / mean_limb_length[
        None
    ]  # (T, J)
    stretching = (
        np.clip(stretching, -max_strech, max_strech) / max_strech
    )  # (T, J) [-1, 1]
    if neg_color is None:
        stretching = np.abs(stretching)
        neg_color = pos_color
    return _metric_colors(
        stretching, base_color, pos_color, neg_color, hsv, brightness, contrast
    )


def jitter_color(
    limb_lenghts: np.ndarray,
    mean_limb_length: np.ndarray,
    max_jitter: float,
    base_color: np.ndarray,
    pos_color: np.ndarray,
    neg_color: np.ndarray | None = None,
    hsv: np.ndarray = np.array([1.0, 1.0, 1.0]),
    brightness: np.ndarray = np.array([1.0]),
    contrast: np.ndarray = np.array([1.0]),
) -> np.ndarray:

    jitter = (limb_lenghts[1:] - limb_lenghts[:-1]) / mean_limb_length[None]  # (T, J)
    jitter = np.clip(jitter, -max_jitter, max_jitter) / max_jitter  # (T, J) [-1, 1]
    if neg_color is None:
        jitter = np.abs(jitter)
        neg_color = pos_color
    return _metric_colors(
        jitter, base_color, pos_color, neg_color, hsv, brightness, contrast
    )
