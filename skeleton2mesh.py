from pathlib import Path
import argparse
import torch
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.datasets import get_skeleton_info
from src.mesh.io import save_obj, to_mesh
from src.skeleton.io import load_skeleton, remove_hip_motion
from src.skeleton.conversions import convert2humanml
from src.skeleton2mesh.joints2smpl import Skeleton2Obj
from src.skeleton2mesh.smpl2mesh import SMPL2Mesh



def npy_path_to_obj_path(npy_path: Path) -> Path:
    return Path(npy_path.parent / f"{npy_path.stem}_obj")

def get_num_files_in_dir(directory: Path) -> int:
    return len(list(directory.glob("*.obj")))

def get_humanml_motion(npy_file, skeleton, unit, inplace=True):
    motion = load_skeleton(npy_file, unit=unit)

    if inplace:
        motion = remove_hip_motion(motion)
    humanml_motion = convert2humanml(
        motion, skeleton.LANDMARKS, skeleton.TO_HUMANML_NAMES
    )
    return humanml_motion

def get_preprocessed_motion(npy_file, skeleton, config, device):
    humanml_motion = torch.from_numpy(get_humanml_motion(npy_file, skeleton, config["unit"], inplace=config.get("inplace", True))).to(device).float()
    return humanml_motion

def save_mesh(vertices, faces, npy_file):
    results_dir = npy_path_to_obj_path(npy_file)
    results_dir.mkdir(parents=True, exist_ok=True)
    for frame_i in tqdm(range(vertices.shape[-1])):
        file_path = results_dir / f"frame{frame_i:03d}.obj"
        mesh = to_mesh(vertices[..., frame_i], faces)
        save_obj(mesh, file_path)
    print(f"Saved obj files to [{results_dir}]")

def if_motion_is_converted(npy_file, motion):
    # check if all the files already exist
    num_files = get_num_files_in_dir(npy_path_to_obj_path(npy_file))
    if num_files == motion.shape[0]:
        print(
            f"Skipping [{npy_file}] as all [{num_files}] files already exist."
        )
        return True
    return False

def select_prediction_to_convert(config, pred_files, motions):
    if config.get("redo", False):
        return motions, pred_files
    else:
        valid_indices = [i for i, npy_file in enumerate(pred_files) if not if_motion_is_converted(npy_file, motions[i])]
        pred_motions = [motions[i] for i in valid_indices]
        pred_files = [pred_files[i] for i in valid_indices]
        if len(pred_motions) == 0:
            return [], []
        pred_motions = torch.stack(pred_motions, dim=0)
    return pred_motions, pred_files


def main(config: dict):
    dir_path = Path(config["directory"])
    assert dir_path.exists() and dir_path.is_dir(), f"Directory {dir_path} does not exist or is not a directory"
    if torch.cuda.is_available():
        device = config.get("device", "cuda")
    else:
        device = "cpu"
    skeleton2obj = Skeleton2Obj(
        device=device, num_smplify_iters=config["num_smplify_iters"], 
        smpl_model_dir="./body_models/", gmm_model_dir="./body_models/joints2smpl/",
    )
    smpl2mesh = SMPL2Mesh(device)

    skeleton = get_skeleton_info('amass')

    # get all the npy files in the directory
    npy_files = sorted(list(Path(config["directory"]).rglob("*.npy")))
    obs_file = [f  for f in npy_files if 'obs' in str(f)][0]
    pred_files = [f for f in npy_files if f != obs_file]


    output_file = Path(config["directory"]) / "obs_as_optimized_smpl.npz"
    
    if output_file.exists():
        # Load the dictionary of arrays from the npz file
        loaded_data = np.load(output_file, allow_pickle=True)
        rot_motions_obs = torch.from_numpy(loaded_data["rot_motions_obs"]).to(device).float()
        smpl_dict_obs = {key: torch.from_numpy(val).to(device).float() for key, val in loaded_data["smpl_dict_obs"].item().items()}
        print("Loaded observation data from npz file.")
        humanml_motion_obs = get_preprocessed_motion(obs_file, skeleton, config, device)
    if not output_file.exists() and config.get("save_obs_smpl_data", True):
        humanml_motion_obs = get_preprocessed_motion(obs_file, skeleton, config, device)
        rot_motions_obs, smpl_dict_obs = skeleton2obj.convert_motion_2smpl(humanml_motion_obs, hmp=False, fix_betas=False, if_smooth_head_motion=config.get("if_smooth_head_motion", True), num_frames=len(humanml_motion_obs))
        # Save the dictionary of arrays to a file
        output_dict = {
            "rot_motions_obs": rot_motions_obs.cpu().numpy(),
            "smpl_dict_obs": {key: value.cpu().numpy() for key, value in smpl_dict_obs.items()},
        }
        np.savez(output_file, **output_dict)
        print(f"Saved observation data to [{output_file}]")

    if not config.get("redo", False):
        # check if all the files already exist
        num_files = get_num_files_in_dir(npy_path_to_obj_path(obs_file))
        if num_files == humanml_motion_obs.shape[0]:
            print(
                f"Skipping [{obs_file}] as all [{num_files}] files already exist."
            )
        else:
            vertices_obs, faces_obs = smpl2mesh.convert_smpl_to_mesh(rot_motions_obs, humanml_motion_obs)
            save_mesh(vertices_obs, faces_obs, obs_file)
    else:
        vertices_obs, faces_obs = smpl2mesh.convert_smpl_to_mesh(rot_motions_obs, humanml_motion_obs)
        save_mesh(vertices_obs, faces_obs, obs_file)

    pred_motions = torch.stack([get_preprocessed_motion(npy_file, skeleton, config, device) for npy_file in pred_files], dim=0)
    pred_motions, pred_files = select_prediction_to_convert(config, pred_files, pred_motions)
    if len(pred_motions) == 0:
        print("All prediction files have been converted already. Exiting.")
        return
    num_frames = pred_motions.shape[-3]
    pred_motions = pred_motions.view(-1, 22, 3).to(device).float()

    init_params = {}
    if config.get('optimize_fut_with_obs_init', True):

        init_params["betas"] = smpl_dict_obs["betas"][-1].unsqueeze(0).expand(pred_motions.shape[0], -1).to(device)
        init_params["pose"] = smpl_dict_obs["pose"][-1].unsqueeze(0).expand(pred_motions.shape[0], -1, -1).view(pred_motions.shape[0], -1).to(device)
        init_params["cam"] = smpl_dict_obs["cam"].to(device)

        rot_motions, smpl_dict = skeleton2obj.convert_motion_2smpl(pred_motions, hmp=True, init_params=init_params, fix_betas=True, 
                                                                   if_smooth_head_motion=config.get("if_smooth_head_motion", True), num_frames=num_frames)
    else:
        rot_motions, smpl_dict = skeleton2obj.convert_motion_2smpl(pred_motions, hmp=False, if_smooth_head_motion=config.get("if_smooth_head_motion", True), num_frames=num_frames)

    
    vertices, faces = smpl2mesh.convert_smpl_to_mesh(rot_motions, pred_motions)
    vertices = vertices.reshape(*vertices.shape[:2], len(pred_files), -1)
    for v, npy_file in zip(np.moveaxis(vertices, 2, 0), pred_files):
        save_mesh(v, faces, npy_file)



if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description='Convert skeleton motion to SMPL mesh')
        parser.add_argument('--directory', type=str, required=True,
                            help='Directory containing .npy files to process')
        parser.add_argument('--unit', type=str, default='m',
                            choices=['m', 'mm', 'cm'],
                            help='Unit of measurement in npy files (default: m)')
        parser.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'],
                            help='Device to use for computation (default: cuda)')
        parser.add_argument('--num_smplify_iters', type=int, default=25,
                            help='Number of SMPLify iterations (default: 25 if the skeleton motion does not have strong artifacts. Otherwise up to 150)')
        parser.add_argument('--redo', action='store_true', default=False,
                            help='Overwrite existing files (default: False)')
        parser.add_argument('--save_obs_smpl_data', action='store_true', default=True,
                            help='Save observation SMPL data (default: True)')
        parser.add_argument('--optimize_fut_with_obs_init', action='store_true', default=True,
                            help='Optimize future with observation initialization (default: True)')
        parser.add_argument('--if_smooth_head_motion', action='store_true', default=True,
                            help='Smooth head motion (default: True)')
        
        args = parser.parse_args()
        config = vars(args)
        
        return config
    
    config = parse_args()
    main(config)