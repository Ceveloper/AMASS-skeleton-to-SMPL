from dataclasses import dataclass

from .common import SkeletonBaseClass


FPS=50

LANDMARKS: list = [
    "Hip",  # 0
    "RHip",  # 1
    "RKnee",  # 2
    "RAnkle",  # 3
    "LHip",  # 4
    "LKnee",  # 5
    "LAnkle",  # 6
    "Torso",  # 7
    "Neck",  # 8
    "Nose",  # 9
    "Head",  # 10
    "LShoulder",  # 11
    "LElbow",  # 12
    "LWrist",  # 13
    "RShoulder",  # 14
    "RElbow",  # 15
    "RWrist",  # 16
]

SKELETON_NAMES = [
    ["LWrist", "LElbow", "LShoulder", "Neck"],
    ["RWrist", "RElbow", "RShoulder", "Neck"],
    ["LAnkle", "LKnee", "LHip", "Hip"],
    ["RAnkle", "RKnee", "RHip", "Hip"],
    ["Hip", "Neck", "Nose", "Head"],
]

SKELETON = [
    [LANDMARKS.index(skeleton_name) for skeleton_name in sub_skeleton_names]
    for sub_skeleton_names in SKELETON_NAMES
]

ROOT_INDEX = LANDMARKS.index("Hip")

MIRRORED_LANDMARKS = {
    "Nose": "Nose",  # 0
    "LEye": "REye",  # 1
    "REye": "LEye",  # 2
    "LEar": "REar",  # 3
    "REar": "LEar",  # 4
    "LShoulder": "RShoulder",  # 5
    "RShoulder": "LShoulder",  # 6
    "LElbow": "RElbow",  # 7
    "RElbow": "LElbow",  # 8
    "LWrist": "RWrist",  # 9
    "RWrist": "LWrist",  # 10
    "LHip": "RHip",  # 11
    "RHip": "LHip",  # 12
    "LKnee": "RKnee",  # 13
    "RKnee": "LKnee",  # 14
    "LAnkle": "RAnkle",  # 15
    "RAnkle": "LAnkle",  # 16
    "Head": "Head",  # 17
    "Neck": "Neck",  # 18
    "Hip": "Hip",  # 19
    "LBigToe": "RBigToe",  # 20
    "RBigToe": "LBigToe",  # 21
    "LSmallToe": "RSmallToe",  # 22
    "RSmallToe": "LSmallToe",  # 23
    "LHeel": "RHeel",  # 24
    "RHeel": "LHeel",  # 25
    "Torso": "Torso",
}

MIRRORED_INDICES = [
    LANDMARKS.index(MIRRORED_LANDMARKS[landmark]) for landmark in LANDMARKS
]

TO_HUMANML_NAMES = [
    ("pelvis", "Hip"),
    ("right_hip", "RHip"),
    ("right_knee", "RKnee"),
    ("right_ankle", "RAnkle"),
    ("left_hip", "LHip"),
    ("left_knee", "LKnee"),
    ("left_ankle", "LAnkle"),
    ("spine1", "Torso"),
    ("neck", "Neck"),
    ("head", "Head"),
    ("left_shoulder", "LShoulder"),
    ("left_elbow", "LElbow"),
    ("left_wrist", "LWrist"),
    ("right_shoulder", "RShoulder"),
    ("right_elbow", "RElbow"),
    ("right_wrist", "RWrist"),
    ("left_foot", "LAnkle"),
    ("right_foot", "RAnkle"),
]

LIMBSEQ = [
    [0, 1],
    [0, 4],  # hips
    [1, 2],
    [2, 3],  # right foot
    [4, 5],
    [5, 6],  # left foot
    [0, 7],
    [7, 8],
    [8, 9],
    [9, 10],  # torso and head
    [8, 11],
    [8, 14],  # shoulders
    [11, 12],
    [12, 13],  # left hand
    [14, 15],
    [15, 16],  # right hand
]

PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
PARENTS_COLOR_LIST = ['m', 'r', 'r', 'r', 'l', 'l', 'l', 'm', 'm', 'm', 'm',  'l', 'l',  'l', 'r', 'r', 'r',]

LIMB_POSITIONS = [
    "m",
    "m",
    "r",
    "r",
    "l",
    "l",
    "m",
    "m",
    "m",
    "m",
    "l",
    "r",
    "l",
    "l",
    "r",
    "r",
]


H36M17 = SkeletonBaseClass(
    LANDMARKS,
    SKELETON_NAMES,
    SKELETON,
    ROOT_INDEX,
    MIRRORED_LANDMARKS,
    MIRRORED_INDICES,
    TO_HUMANML_NAMES,
    LIMBSEQ,
    LIMB_POSITIONS,
    PARENTS,
    PARENTS_COLOR_LIST,
    FPS
)
