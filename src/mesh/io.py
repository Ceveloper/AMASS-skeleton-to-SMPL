from pathlib import Path

import numpy as np

from trimesh import Trimesh


def to_mesh(vertices: np.ndarray, faces: np.ndarray) -> Trimesh:
    return Trimesh(vertices=vertices, faces=faces)


def save_obj(mesh: Trimesh, save_path: Path):
    with open(save_path, "w") as fw:
        mesh.export(fw, "obj")
