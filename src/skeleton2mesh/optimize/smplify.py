import os
import sys

import numpy as np
import torch

from datetime import datetime
from datetime import timedelta

from .customloss import (
    body_fitting_loss_3d,
    camera_fitting_loss_3d,
)
from .prior import MaxMixturePrior

import src.skeleton2mesh.optimize.config as config


@torch.no_grad()
def guess_init_3d(model_joints, j3d, joints_category="orig"):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model_joints: SMPL model with pre joints
    :param j3d: 25x3 array of Kinect Joints
    :returns: 3D vector corresponding to the estimated camera translation
    """
    # get the indexed four
    gt_joints = ["RHip", "LHip", "RShoulder", "LShoulder"]
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "orig":
        joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == "AMASS":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint]
                               for joint in gt_joints]
    else:
        print("NO SUCH JOINTS CATEGORY!")

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(
        dim=1
    )
    init_t = sum_init_t / 4.0
    return init_t


# SMPLIfy 3D
class SMPLify3D:
    """Implementation of SMPLify, use 3D joints."""

    def __init__(
        self,
        smplxmodel,
        step_size=1e-2,
        num_iters=100,
        joints_category="orig",
        device=torch.device("cuda:0"),
        GMM_MODEL_DIR="./body_models/joints2smpl/",
    ):

        # Store options
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters

        # GMM pose prior
        self.pose_prior = MaxMixturePrior(
            prior_folder=GMM_MODEL_DIR, num_gaussians=8, dtype=torch.float32
        ).to(device)

        # reLoad SMPL-X model
        self.smpl = smplxmodel

        self.model_faces = smplxmodel.faces_tensor.view(-1)

        # select joint joint_category
        self.joints_category = joints_category

        if joints_category == "orig":
            self.smpl_index = config.full_smpl_idx
            self.corr_index = config.full_smpl_idx
        elif joints_category == "AMASS":
            self.smpl_index = config.amass_smpl_idx
            self.corr_index = config.amass_idx
        else:
            self.smpl_index = None
            self.corr_index = None
            print("NO SUCH JOINTS CATEGORY!")

    # ---- get the man function here ------
    def __call__(self, init_pose, init_betas, init_cam_t, j3d, conf_3d=1.0, fix_betas=0, if_simple_hmp_optimizes=False, num_iters=None):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            j3d: joints 3d aka keypoints
            conf_3d: confidence for 3d joints
                        seq_ind: index of the sequence
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """

        # # # add the mesh inter-section to avoid
        search_tree = None
        pen_distance = None
        filter_faces = None
        self.t0 = datetime.now()

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()
        camera_translation = init_cam_t.clone()

        preserve_pose = init_pose[:, 3:].detach().clone()

        # -------------Step 1: Optimize camera translation and body orientation--------
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        if not if_simple_hmp_optimizes:
            camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        # camera_optimizer = torch.optim.LBFGS(
        #     camera_opt_params,
        #     max_iter=self.num_iters,
        #     lr=self.step_size,
        #     line_search_fn="strong_wolfe",
        # )
        # for i in range(10):

        #     def closure():
        #         camera_optimizer.zero_grad()
        #         smpl_output = self.smpl(
        #             global_orient=global_orient, body_pose=body_pose, betas=betas
        #         )
        #         model_joints = smpl_output.joints
        #         loss = camera_fitting_loss_3d(
        #             model_joints,
        #             camera_translation,
        #             init_cam_t,
        #             j3d,
        #             self.joints_category,
        #         )
        #         loss.backward()
        #         return loss

        #     camera_optimizer.step(closure)

        camera_optimizer = torch.optim.Adam(
            camera_opt_params, lr=self.step_size, betas=(0.9, 0.999)
        )

        for i in range(10):
            smpl_output = self.smpl(
                global_orient=global_orient, body_pose=body_pose, betas=betas
            )
            model_joints = smpl_output.joints

            loss = camera_fitting_loss_3d(
                model_joints[:, self.smpl_index],
                camera_translation,
                init_cam_t,
                j3d[:, self.corr_index],
                self.joints_category,
            )
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        self.t = datetime.now() - self.t0
        self.t0 = datetime.now()
        print(
            f"Step 0: Average time   in seconds: {self.t/timedelta(seconds=1)}")

        # Fix camera translation after optimizing camera
        # --------Step 2: Optimize body joints --------------------------
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        if not if_simple_hmp_optimizes:
            camera_translation.requires_grad = True
        # --- if we use the sequence, fix the shape
        if not fix_betas:
            betas.requires_grad = True
            body_opt_params = [body_pose, betas,
                               global_orient, camera_translation]
        else:
            betas.requires_grad = False
            body_opt_params = [body_pose, global_orient, camera_translation]

        num_iters = self.num_iters if num_iters is None else num_iters

        body_optimizer = torch.optim.LBFGS(
            body_opt_params,
            max_iter=num_iters,
            lr=self.step_size,
            line_search_fn="strong_wolfe",
        )
        for i in range(num_iters):

            def closure():
                body_optimizer.zero_grad()
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices

                loss = body_fitting_loss_3d(
                    body_pose,
                    preserve_pose,
                    betas,
                    model_joints[:, self.smpl_index],
                    camera_translation,
                    j3d[:, self.corr_index],
                    self.pose_prior,
                    joints3d_conf=conf_3d,
                    joint_loss_weight=600.0,
                    pose_preserve_weight=5.0,
                    use_collision=False,
                    model_vertices=model_vertices,
                    model_faces=self.model_faces,
                    search_tree=search_tree,
                    pen_distance=pen_distance,
                    filter_faces=filter_faces,
                )
                loss.backward()
                return loss

            body_optimizer.step(closure)
        # body_optimizer = torch.optim.Adam(
        #         body_opt_params, lr=1.e-4, betas=(0.9, 0.999)
        #     )

        # for i in range(num_iters):
        #     smpl_output = self.smpl(
        #         global_orient=global_orient, body_pose=body_pose, betas=betas
        #     )
        #     model_joints = smpl_output.joints
        #     model_vertices = smpl_output.vertices

        #     loss = body_fitting_loss_3d(
        #         body_pose,
        #         preserve_pose,
        #         betas,
        #         model_joints[:, self.smpl_index],
        #         camera_translation,
        #         j3d[:, self.corr_index],
        #         self.pose_prior,
        #         joints3d_conf=conf_3d,
        #         joint_loss_weight=600.0,
        #         use_collision=False,
        #         model_vertices=model_vertices,
        #         model_faces=self.model_faces,
        #         search_tree=search_tree,
        #         pen_distance=pen_distance,
        #         filter_faces=filter_faces,
        #     )
        #     body_optimizer.zero_grad()
        #     loss.backward()
        #     body_optimizer.step()
        self.t = datetime.now() - self.t0
        self.t0 = datetime.now()
        print(
            f"Step2: Average time   in seconds: {self.t/timedelta(seconds=1)}")
        smpl_output = self.smpl(
            global_orient=global_orient, body_pose=body_pose, betas=betas
        )
        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation
