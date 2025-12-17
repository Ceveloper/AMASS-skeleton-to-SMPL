from dataclasses import dataclass
from .common import SkeletonBaseClass

FPS=50

LANDMARKS = [
    "Hip",  # 0
    "RHip",  # 1
    "RKnee",  # 2
    "RAnkle",  # 3
    "RFoot",  # 4
    "RToes",  # 5
    "LHip",  # 6
    "LKnee",  # 7
    "LAnkle",  # 8
    "LFoot",  # 9
    "LToes",  # 10
    "Torso",  # 11
    "Neck",  # 12
    "Nose",  # 13
    "Head",  # 14
    "LShoulder",  # 15
    "LElbow",  # 16
    "LWrist",  # 17
    "LSmallFinger",  # 18
    "LThumb",  # 19
    "RShoulder",  # 20
    "RElbow",  # 21
    "RWrist",  # 22
    "RSmallFinger",  # 23
    "RThumb",  # 24
]

SKELETON_NAMES = [
    ["LSmallFinger", "LThumb", "LWrist", "LElbow", "LShoulder", "Neck"],
    ["RSmallFinger", "RThumb", "RWrist", "RElbow", "RShoulder", "Neck"],
    ["LFoot", "LToes", "LAnkle", "LKnee", "LHip", "Hip"],
    ["RFoot", "RToes", "RAnkle", "RKnee", "RHip", "Hip"],
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
    "LSmallFinger": "RSmallFinger",  # 26
    "RSmallFinger": "LSmallFinger",  # 27
    "LThumb": "RThumb",  # 27
    "RThumb": "LThumb",  # 27
    "LFoot": "RFoot",  # 28
    "RFoot": "LFoot",  # 28
    "LToes": "RToes",  # 29
    "RToes": "LToes",  # 29
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
    ("left_foot", "LFoot"),
    ("right_foot", "RFoot"),
    ("left_foot", "LToes"),
    ("right_foot", "RToes"),
    ("left_wrist", "LSmallFinger"),
    ("right_wrist", "RSmallFinger"),
    ("left_wrist", "LThumb"),
    ("right_wrist", "RThumb"),
]


LIMBSEQ = [
    [0, 1],
    [0, 6],  # hips
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],  # right foot
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],  # left foot
    [0, 11],
    [11, 12],
    [12, 13],
    [13, 14],  # torso and head
    [12, 15],
    [12, 20],  # shoulders
    [15, 16],
    [16, 17],
    [17, 18],
    [17, 19],  # left hand
    [20, 21],
    [21, 22],
    [22, 23],
    [22, 24],  # right hand
]

# parents = [None] * (max(max(LIMBSEQ))+1)
# parents[0] = -1
# for tup in LIMBSEQ:
#     parents[tup[1]] = tup[0]
PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 12, 15, 16, 17, 17, 12, 20, 21, 22, 22]
# clors_list = ['r' if name[0]=='R' else None for name in LANDMARKS]
# clors_list = ['l' if clors_list[i] is None and name[0]=='L' else clors_list[i] for i,name in enumerate(LANDMARKS)]
PARENTS_COLOR_LIST = ['m', 'r', 'r', 'r', 'r', 'r', 'l', 'l', 'l', 'l', 'l', 'm', 'm', 'm', 'm', 'l', 'l', 'l', 'l', 'l', 'r', 'r', 'r', 'r', 'r']

LIMB_POSITIONS = [
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
    "m",
    "m",
    "m",
    "m",
    "l",
    "r",
    "l",
    "l",
    "l",
    "l",
    "r",
    "r",
    "r",
    "r",
]


H36M25 = SkeletonBaseClass(
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
