from dataclasses import dataclass


FPS=60
from .common import SkeletonBaseClass

LANDMARKS = [
    "pelvis",  # check
    "left_hip",  # check
    "right_hip",  # check
    "spine1",  # check
    "left_knee",  # check
    "right_knee",  # check
    "spine2",
    "left_ankle",  # check
    "right_ankle",  # check
    "spine3",
    "left_foot",  # check, but wrong
    "right_foot",  # check, but wrong
    "neck",  # check
    "left_collar",
    "right_collar",
    "head",  # check
    "left_shoulder",  # check
    "right_shoulder",  # check
    "left_elbow",  # check
    "right_elbow",  # check
    "left_wrist",  # check
    "right_wrist",  # check
]
# ['GlobalHip', 'RHip', 'LHip', 'Spine1', 'RKnee', 'LKnee', 'Spine3', 'RHeel', 'LHeel', 'Neck', 'RFoot', 'LFoot', 'BMN', 'RSI', 'LSI', 'Head', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist']

SKELETON_NAMES = [
    ["pelvis", "right_hip", "right_knee", "right_ankle", "right_foot"],
    ["pelvis", "left_hip", "left_knee", "left_ankle", "left_foot"],
    ["pelvis", "spine1", "spine2", "spine3", "neck", "head"],
    ["spine3", "right_collar", "right_shoulder", "right_elbow", "right_wrist"],
    ["spine3", "left_collar", "left_shoulder", "left_elbow", "left_wrist"],
]

SKELETON = [[LANDMARKS.index(name) for name in names] for names in SKELETON_NAMES]

ROOT_INDEX = LANDMARKS.index("pelvis")

MIRRORED_LANDMARKS = {
    "pelvis": "pelvis",
    "left_hip": "right_hip",
    "right_hip": "left_hip",
    "spine1": "spine1",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
    "spine2": "spine2",
    "left_ankle": "right_ankle",
    "right_ankle": "left_ankle",
    "spine3": "spine3",
    "left_foot": "right_foot",
    "right_foot": "left_foot",
    "neck": "neck",
    "left_collar": "right_collar",
    "right_collar": "left_collar",
    "head": "head",
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_elbow": "right_elbow",
    "right_elbow": "left_elbow",
    "left_wrist": "right_wrist",
    "right_wrist": "left_wrist",
}

MIRRORED_INDICES = [
    LANDMARKS.index(MIRRORED_LANDMARKS[landmark]) for landmark in LANDMARKS
]

TO_HUMANML_NAMES = [
    ("pelvis", "pelvis"),
    ("right_hip", "right_hip"),
    ("right_knee", "right_knee"),
    ("right_ankle", "right_ankle"),
    ("right_foot", "right_foot"),
    ("left_hip", "left_hip"),
    ("left_knee", "left_knee"),
    ("left_ankle", "left_ankle"),
    ("left_foot", "left_foot"),
    ("spine1", "spine1"),
    ("spine2", "spine2"),
    ("spine3", "spine3"),
    ("neck", "neck"),
    ("head", "head"),
    ("left_collar", "left_collar"),
    ("right_collar", "right_collar"),
    ("left_shoulder", "left_shoulder"),
    ("right_shoulder", "right_shoulder"),
    ("left_elbow", "left_elbow"),
    ("right_elbow", "right_elbow"),
    ("left_wrist", "left_wrist"),
    ("right_wrist", "right_wrist"),
]

LIMBSEQ = [
    [0, 3],
    [3, 6],
    [6, 9],
    [9, 12],
    [12, 15],  # body
    [9, 14],
    [14, 17],
    [17, 19],
    [19, 21],  # right arm
    [9, 13],
    [13, 16],
    [16, 18],
    [18, 20],  # left arm
    [0, 2],
    [2, 5],
    [5, 8],
    [8, 11],  # right leg
    [0, 1],
    [1, 4],
    [4, 7],
    [7, 10],  # left leg
]

PARENTS= [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
PARENTS_COLOR_LIST = ["m", 'r', 'l', "m", 'r', 'l', "m", 'r', 'l', "m", 'r', 'l', "m", 'r', 'l', "m", 'r', 'l', 'r', 'l', 'r', 'l']

LIMB_POSITIONS = [
    "m",
    "m",
    "m",
    "m",
    "m",
    "r",
    "r",
    "r",
    "r",
    "l",
    "l",
    "l",
    "l",
    "r",
    "r",
    "r",
    "r",
    "l",
    "l",
    "l",
    "l",
]


Amass = SkeletonBaseClass(
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


