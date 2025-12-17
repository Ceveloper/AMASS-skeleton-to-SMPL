from . import amass, h36m17, h36m25
from .common import LANDMARKS, SkeletonBaseClass


def get_skeleton_info(name) -> SkeletonBaseClass:
    """
    Get conversion names.
    Args:
        name: Skeleton type.
    Returns:
        joint_to_humanml_names: Conversion names.
    """
    if name in ["h36m17"] or name == "h36m":
        return h36m17.H36M17
    elif name in ["h36m25"]:
        return h36m25.H36M25
    elif name in ["amass"]:
        return amass.Amass
    raise ValueError(f"Unsupported skeleton type [{name}]")
