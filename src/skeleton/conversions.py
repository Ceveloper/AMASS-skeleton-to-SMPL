import numpy as np

from src.datasets import LANDMARKS


def convert2humanml(
    motion: np.ndarray,
    dataset_landmarks: list[str],
    joint_to_humanml_names: list[tuple[str, str]],
) -> np.ndarray:
    """
    Convert skeletons to human ml.
    Args:
        skeletons: Skeletons.
        dataset_landmarks: Landmarks of the dataset.
        joint_to_humanml_names: Conversion names.
    Returns:
        new_motion: Skeletons in human ml.
    """
    new_motion = np.zeros((*motion.shape[:-2], len(LANDMARKS), 3))
    indices = np.array(
        [
            (LANDMARKS.index(hml_name), dataset_landmarks.index(nba_name))
            for hml_name, nba_name in joint_to_humanml_names
        ]
    )
    new_motion[..., indices[:, 0], :] = motion[..., indices[:, 1], :]

    # Infer missing landmarks
    covered_landmarks = [hml_landmark for hml_landmark,
                         _ in joint_to_humanml_names]
    root_index, left_hip_index, right_hip_index = (
        LANDMARKS.index("pelvis"),
        LANDMARKS.index("left_hip"),
        LANDMARKS.index("right_hip"),
    )
    if "pelvis" not in covered_landmarks:
        assert (
            "left_hip" in covered_landmarks and "right_hip" in covered_landmarks
        ), "pelvis could not be inferred from left_hip and right_hip"
        new_motion[..., root_index, :] = (
            new_motion[..., left_hip_index, :] +
            new_motion[..., right_hip_index, :]
        ) / 2

    neck_index, left_shoulder_index, right_shoulder_index = (
        LANDMARKS.index("neck"),
        LANDMARKS.index("left_shoulder"),
        LANDMARKS.index("right_shoulder"),
    )
    if "neck" not in covered_landmarks:
        assert (
            "left_shoulder" in covered_landmarks
            and "right_shoulder" in covered_landmarks
        ), "neck could not be inferred from left_shoulder and right_shoulder"
        new_motion[..., neck_index, :] = (
            new_motion[..., left_shoulder_index, :]
            + new_motion[..., right_shoulder_index, :]
        ) / 2

    spine_1_index, spine_2_index, spine_3_index = (
        LANDMARKS.index("spine1"),
        LANDMARKS.index("spine2"),
        LANDMARKS.index("spine3"),
    )
    if "spine1" not in covered_landmarks:
        new_motion[..., spine_1_index, :] = (
            0.75 * new_motion[..., root_index, :]
            + 0.25 * new_motion[..., neck_index, :]
        )
    if "spine2" not in covered_landmarks:
        new_motion[..., spine_2_index, :] = (
            0.5 * new_motion[..., root_index, :] +
            0.5 * new_motion[..., neck_index, :]
        )
    if "spine3" not in covered_landmarks:
        new_motion[..., spine_3_index, :] = (
            0.25 * new_motion[..., root_index, :]
            + 0.75 * new_motion[..., neck_index, :]
        )

    left_collar_index, right_collar_index = LANDMARKS.index(
        "left_collar"
    ), LANDMARKS.index("right_collar")
    if "left_collar" not in covered_landmarks:
        assert (
            "left_shoulder" in covered_landmarks
        ), "left_collar could not be inferred from left_shoulder"
        new_motion[..., left_collar_index, :] = (
            0.25 * new_motion[..., left_shoulder_index, :]
            + 0.75 * new_motion[..., neck_index, :]
        )
    if "right_collar" not in covered_landmarks:
        assert (
            "right_shoulder" in covered_landmarks
        ), "right_collar could not be inferred from right_shoulder"
        new_motion[..., right_collar_index, :] = (
            0.25 * new_motion[..., right_shoulder_index, :]
            + 0.75 * new_motion[..., neck_index, :]
        )

    return new_motion
