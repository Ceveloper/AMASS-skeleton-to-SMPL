import torch

from .smpl_utils.rotation2xyz import Rotation2xyz
from .smpl_utils.rotation_conversions import matrix_to_rotation_6d, axis_angle_to_matrix

class SMPL2Mesh:
    def __init__(
        self,
        device: str = "cpu",
    ):
        self.rot2xyz = Rotation2xyz(device=device)
        self.faces = self.rot2xyz.smpl_model.faces


    def convert_smpl_to_mesh(self, new_opt_pose, keypoints_3d):
        batch_size = keypoints_3d.shape[0]

        thetas = new_opt_pose.reshape(batch_size, 24, 3)
        thetas = matrix_to_rotation_6d(
            axis_angle_to_matrix(thetas)
        )  # [bs, 24, 6]
        # root_loc = torch.tensor(keypoints_3d[:, 0])  # [bs, 3]
        root_loc = keypoints_3d[:, 0].clone()  # [bs, 3]
        root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-1).unsqueeze(
            1
        )  # [bs, 1, 6]
        thetas = torch.cat([thetas, root_loc], dim=1).permute(1, 2, 0)  # [25, 6, 196]
        rot_motions = thetas.detach()
        vertices = self.rot2xyz(
            rot_motions,
            mask=None,
            pose_rep="rot6d",
            translation=True,
            glob=True,
            jointstype="vertices",
            vertstrans=True,
        )

        return vertices.cpu().numpy(), self.faces