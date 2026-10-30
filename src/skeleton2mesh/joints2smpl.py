import numpy as np
import smplx
import torch
from scipy.ndimage import gaussian_filter1d

from .optimize.smplify import SMPLify3D


class joints2smpl:
    def __init__(
        self,
        device,
        num_smplify_iters=150,
        SMPL_MODEL_DIR="./body_models/",
        GMM_MODEL_DIR="./body_models/joints2smpl/",
    ):
        self.device = torch.device(device)
        # self.batch_size = num_frames
        self.num_joints = 22  # for HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = num_smplify_iters
        self.fix_foot = False
        # smplmodel = smplx.create( SMPL_MODEL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=self.batch_size).to(self.device)
        smplmodel = smplx.create(
            SMPL_MODEL_DIR, model_type="smpl", gender="female", ext="pkl"
        ).to(self.device)

        # # #-------------initialize SMPLify
        # self.smplify = SMPLify3D(smplxmodel=smplmodel, batch_size=self.batch_size, joints_category=self.joint_category, num_iters=self.num_smplify_iters, device=self.device)
        self.smplify = SMPLify3D(
            smplxmodel=smplmodel,
            # batch_size=self.batch_size,
            joints_category=self.joint_category,
            num_iters=self.num_smplify_iters,
            device=self.device,
            GMM_MODEL_DIR=GMM_MODEL_DIR
        )

    def joint2smpl(self, input_joints, init_params=None, hmp=True, fix_betas=False):
        batch_size = input_joints.shape[0]

        _smplify = self.smplify  # if init_params is None else self.smplify_fast
        pred_pose = torch.zeros(batch_size, 72).to(self.device)
        pred_betas = torch.zeros(1, 10).expand(batch_size, 10).to(self.device)
        pred_cam_t = torch.zeros(1, 3).expand(batch_size, 3).to(self.device)
        keypoints_3d = torch.Tensor(input_joints).to(self.device).float()

        if init_params is None:
            pred_betas = (
                # torch.from_numpy(self.file["shape"][:])
                torch.tensor([1.47646511e+00,  4.79749501e-01, -6.36047006e-01, -1.52980864e+00,
                              -1.11884427e+00, -5.40487289e-01,  3.93005997e-01, -1.88832569e+00,
                              -2.78680950e-01, -5.49529344e-02])
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .float()
                .to(self.device)
            )
            pred_pose = (torch.tensor([0.4531,  0.3044,  0.2968, -0.2239,  0.0174,  0.0925, -0.2378, -0.0465,
                                       -0.0786,  0.2782,  0.0141,  0.0138,  0.4328, -0.0629, -0.0961,  0.5043,
                                       0.0035,  0.0610,  0.0230, -0.0317,  0.0058,  0.0070,  0.1317, -0.0544,
                                       -0.0589, -0.1752,  0.1355,  0.0134, -0.0037,  0.0089, -0.2093,  0.1600,
                                       0.1092, -0.0387,  0.0824, -0.2041, -0.0056, -0.0075, -0.0035, -0.0237,
                                       -0.1248, -0.2736, -0.0459,  0.1991,  0.2373,  0.0667, -0.0405,  0.0329,
                                       0.0536, -0.2914, -0.6969,  0.0559,  0.2858,  0.6525,  0.1222, -0.9116,
                                       0.2383, -0.0366,  0.9237, -0.2554, -0.0657, -0.1045,  0.0501, -0.0388,
                                       0.0909, -0.0707, -0.1437, -0.0590, -0.1801, -0.0875,  0.1093,  0.2009])
                         .unsqueeze(0)
                         .repeat(batch_size, 1)
                         .float()
                         .to(self.device)
                         )
            pred_cam_t = torch.Tensor(
                [0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)
        else:
            pred_betas = init_params["betas"]
            pred_pose = init_params["pose"]
            pred_cam_t = init_params["cam"]

        if self.joint_category == "AMASS":
            confidence_input = torch.ones(self.num_joints)
            # make sure the foot and ankle
            if self.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            print("Such category not settle down!")

        (
            new_opt_vertices,
            new_opt_joints,
            new_opt_pose,
            new_opt_betas,
            new_opt_cam_t,
        ) = _smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(self.device),
            if_simple_hmp_optimizes=hmp,
            fix_betas=fix_betas,
        )

        # thetas = new_opt_pose.reshape(batch_size, 24, 3)
        # thetas = matrix_to_rotation_6d(
        #     axis_angle_to_matrix(thetas)
        # )  # [bs, 24, 6]
        # # root_loc = torch.tensor(keypoints_3d[:, 0])  # [bs, 3]
        # root_loc = keypoints_3d[:, 0].clone()  # [bs, 3]
        # root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-1).unsqueeze(
        #     1
        # )  # [bs, 1, 6]
        # thetas = torch.cat([thetas, root_loc], dim=1).permute(1, 2, 0)  # [25, 6, 196]
        # thetas = thetas.clone().detach()

        return new_opt_pose, {
            "pose": new_opt_joints[..., :24, :].clone().detach(),
            "betas": new_opt_betas.clone().detach(),
            "cam": new_opt_cam_t.clone().detach(),
        }


def smooth_head_motion(rot_motions, num_frames):

    thetas = rot_motions
    thetas = thetas.view(-1, num_frames, 72).cpu().detach().numpy()
    sigma = 20.0
    for param_idx in range(36, 40):
        thetas[..., param_idx] = gaussian_filter1d(
            thetas[..., param_idx], sigma=sigma, axis=-1)
    for param_idx in range(45, 49):
        thetas[..., param_idx] = gaussian_filter1d(
            thetas[..., param_idx], sigma=sigma, axis=-1)
    thetas = torch.from_numpy(thetas).reshape(-1, 24, 3).to(rot_motions.device)
    rot_motions = thetas
    return rot_motions


class Skeleton2Obj:
    def __init__(
        self,
        device: str = "cpu",
        num_smplify_iters=150,
        smpl_model_dir="./body_models/",
        gmm_model_dir="./body_models/joints2smpl/",
    ):
        self.simplify = joints2smpl(
            device=device,
            num_smplify_iters=num_smplify_iters,
            SMPL_MODEL_DIR=smpl_model_dir,
            GMM_MODEL_DIR=gmm_model_dir,
        )

    def convert_motion_2smpl(self, motion, hmp=True, init_params=None, fix_betas=False, if_smooth_head_motion=True, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        new_opt_rot, smpl_dict = self.simplify.joint2smpl(
            motion, hmp=hmp, init_params=init_params, fix_betas=fix_betas)
        if if_smooth_head_motion:
            new_opt_rot = smooth_head_motion(new_opt_rot, **kwargs)
        return new_opt_rot, smpl_dict
