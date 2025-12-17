
import argparse
import glob
import shutil
import numpy as np
import cv2
from vedo import Mesh, Plane, Plotter, screenshot
import imageio.v2 as imageio
import re
import os
from PIL import Image
from tqdm import tqdm

screenshot_scale = 3

_green = [0.4, 0.7, 0.5]
_dark_blue = [0.03, 0.4, 0.7]

cam = dict(
    position=(-3.85922, -4.78140, 2.689328),
    focal_point=(0.192544, 4.77379e-3, 0.0127248),
    viewup=(0.0724348, 0.109097, 0.991388),
    distance=5.25119,
    clipping_range=(4.38228, 6.36775),
)


def get_mesh(mesh_name, color='darksalmon', mesh_opacity=1.0, mesh_lighting='default', rotation=[0, 0, 0], offset=[0, 0, 0], scale=0):
    offset = np.array(offset)
    # styles = ['default', 'metallic', 'plastic', 'shiny', 'glossy', 'ambient', 'off']
    mesh = Mesh(mesh_name, c=color, alpha=mesh_opacity).lighting(
        mesh_lighting).linewidth(0)
    if scale == 0:
        bnd = np.array(mesh.bounds())
        scale = 1 / max(bnd[1]-bnd[0], bnd[3]-bnd[2], bnd[5]-bnd[4])
    mesh.scale(scale)
    mesh.pos(offset)
    mesh.rotate_x(rotation[0])
    mesh.rotate_y(rotation[1])
    mesh.rotate_z(rotation[2])

    return mesh, scale


def save_png_rm_bg(filename, im, bg_color_min=[0, 255, 100], bg_color_max=None, rgb=False):
    # https://stackoverflow.com/questions/72062001/remove-everything-of-a-specific-color-with-a-color-variation-tolerance-from-an
    # for transperancy:
    # https://stackoverflow.com/questions/55582117/efficiently-converting-color-to-transparency-in-python

    if bg_color_max is None:
        bg_color_max = bg_color_min

    # Define lower and upper limits of our blue
    min = np.array(bg_color_min, np.uint8)
    max = np.array(bg_color_max, np.uint8)

    # Go to HSV colourspace and get mask of blue pixels
    mask = cv2.inRange(im, min, max)

    # create alpha channel
    alpha = (im[:, :, 0] * 0 + 255)

    # Make all pixels in mask white
    alpha[mask > 0] = 0

    # add alpha channel
    if rgb:
        im = cv2.merge((im[:, :, 0], im[:, :, 1], im[:, :, 2], alpha))
    else:
        im = cv2.merge((im[:, :, 2], im[:, :, 1], im[:, :, 0], alpha))

    # crop to visible part
    mask = im[:, :, 3] != 0.
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # write to file
    cv2.imwrite(filename, im)


def get_mesh_shadow(mesh, offset=[0, 0, 0], plane_normal=(0, 0, 1), direction=[0.1, -1.8, 3]):
    shadow = []
    shad_col = np.array([0.8, 0.8, 0.8])
    min_z = mesh.vertices.min(axis=0)[2]
    # mesh.add_shadow(plane='z', point=min_z, alpha=1, c=shad_col, culling=0.9,)
    plane = Plane(pos=np.array(
        [0, 0, min_z]) + np.array(offset), normal=plane_normal, s=[7, 7]).alpha(0.2)
    shad = mesh.clone().project_on_plane(plane, direction=-
                                         np.array(direction) + np.array(offset))
    shad.c(shad_col).alpha(1).lighting("off").use_bounds(False)
    shadow = shad
    return shadow


def create_visual(mesh_files, color='coral', dest_name="", x=0, y=0, z=0, type='mesh', mesh_lighting='plastic', shadow=False):
    meshes, scales = [], []
    bg_col = [255, 255, 255]
    plotter = Plotter(bg=[255, 255, 255], offscreen=True)

    frame_files = []
    for file in tqdm(mesh_files):
        mesh, scale = get_mesh(file, color=color, rotation=[
                               x, y, z], mesh_lighting=mesh_lighting, scale=0.6)
        plotter.clear()
        if shadow:
            shadow = get_mesh_shadow(mesh)
            plotter.show(mesh, shadow, __doc__, interactive=False, camera=cam, resetcam=True, zoom=2)
        else:
            plotter.show(mesh, __doc__, interactive=False,
                        camera=cam, resetcam=True, zoom=2)
        frame_path = file[:-4] + ".png"
        frame_files.append(frame_path)

        screenshot(frame_path, scale=1)
        save_png_rm_bg(
            file[:-4]+".png", plotter.screenshot(asarray=True), bg_color_min=bg_col)

    # Compile saved frames into a GIF
    create_transparent_gif(mesh_files, dest_name)


def create_transparent_gif(mesh_files, dest_name):
    import subprocess
    frames_folder = mesh_files[0].split('frame')[0]
    subprocess.run(["ffmpeg", "-y", "-framerate", "60", "-i", frames_folder+"frame%03d.png",
                   "-vf", "palettegen=reserve_transparent=1", frames_folder+"palette.png"])
    subprocess.run(["ffmpeg", "-framerate", "60", "-i", frames_folder+"frame%03d.png", "-i", frames_folder +
                   "palette.png", "-lavfi", "paletteuse=alpha_threshold=128", "-gifflags", "-offsetting", dest_name + ".gif"])

def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)


def main(parent_folder, dest_dir, if_concat_obs_in_gif, x_rot, y_rot, z_rot, redo, shadow=True):
    file_type = '.obj'  # 'mesh.ply'

    if dest_dir is None:
        dest_dir = os.path.join(parent_folder, 'gif_60fps')
    os.makedirs(dest_dir, exist_ok=True)

    meshfolders = sorted([f for f in os.listdir(parent_folder) if os.path.isdir(
        os.path.join(parent_folder, f)) and 'obj' in f and 'obj_img' not in f])

    # Create transparent pngs and gifs of every sequence
    for f in meshfolders:
        folder = os.path.join(parent_folder, f)
        dest_name = os.path.join(dest_dir, f)
        if os.path.exists(dest_name + ".gif") and not redo:
            continue

        mesh_files = [os.path.join(folder, f) for f in os.listdir(
            folder) if f.endswith(file_type)]
        mesh_files = sort_list(mesh_files)

        if 'obs' in f:
            create_visual(mesh_files, _green, dest_name, x_rot, y_rot, z_rot, type='mesh',
                          mesh_lighting='plastic', shadow=shadow)
        else:
            create_visual(mesh_files, _dark_blue, dest_name, x_rot, y_rot, z_rot,
                          type='mesh', mesh_lighting='plastic', shadow=shadow)

    if if_concat_obs_in_gif:
        dest_dir = dest_dir +  '_concat_obs'
        os.makedirs(dest_dir, exist_ok=True)
        # Find obs subfolder
        obsmeshfolders = sorted([f for f in os.listdir(parent_folder) if os.path.isdir(
            os.path.join(parent_folder, f)) and 'obj' in f])
        obs_subfolder = [sf for sf in obsmeshfolders if 'obs' in sf]
        if obs_subfolder:
            obs_subfolder = obs_subfolder[0]
            obs_folder = os.path.join(parent_folder, obs_subfolder)
            obs_pngs = sort_list([os.path.join(obs_folder, f)
                                  for f in os.listdir(obs_folder) if f.endswith('.png') and 'palette' not in f])

        meshfolders = sorted([f for f in os.listdir(parent_folder) if os.path.isdir(
            os.path.join(parent_folder, f)) and 'obj'  in f])
        temp_dir = os.path.join(parent_folder, 'temp_dir')
        os.makedirs(temp_dir, exist_ok=True)
        for f in meshfolders:
            if 'obs' in f:
                continue
            dest_path = os.path.join(dest_dir, 'obs+' + f + '.gif')
            if os.path.exists(dest_path) and not redo:
                continue
            # Copy obs frames
            for i, obs_frame in enumerate(obs_pngs):
                shutil.copy(obs_frame, os.path.join(
                    temp_dir, f'frame{i:03d}.png'))

            # Copy current f's frames, renaming indices
            f_folder = os.path.join(parent_folder, f)
            f_pngs = sort_list([os.path.join(f_folder, fn)
                                for fn in os.listdir(f_folder) if fn.endswith('.png')  and 'palette' not in fn])
            offset = len(obs_pngs)
            for j, f_frame in enumerate(f_pngs):
                shutil.copy(f_frame, os.path.join(
                    temp_dir, f'frame{j+offset:03d}.png'))

            # Run create_transparent_gif in temp_dir
            temp_frames = sort_list(
                glob.glob(os.path.join(temp_dir, 'frame*.png')))
            temp_dest_name = os.path.join(temp_dir, f)
            create_transparent_gif(temp_frames, temp_dest_name)

            # Move gif to pastfuture_gif_dir
            gif_path = temp_dest_name + '.gif'
            if os.path.exists(gif_path):
                shutil.move(gif_path, dest_path)

            # Remove all future .png from temp_dir
            for png_file in glob.glob(os.path.join(temp_dir, 'frame*.png')):
                match = re.search(r'frame(\d+)\.png',
                                  os.path.basename(png_file))
                if match and int(match.group(1)) >= offset:
                    os.remove(png_file)

        shutil.rmtree(temp_dir)
        

    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parent_folder', type=str,
                        default="/usr/wiss/curreli/work/my_modules/Skeleton2Mesh/final_output/hmp/visuals_50samples/amass/SkeletonDiffusion/test_presentation/86_Transitions_smoothed2", help="Path to the parent folder containing mesh folders.")
    parser.add_argument('-d', '--dest_dir', type=str, default=None,
                        help="Path to the destination folder for saving gifs. If None, it will be created inside parent_folder.")
    parser.add_argument('--redo', action='store_true', default=False,
                            help='Overwrite existing files (default: False)')
    parser.add_argument('--if_concat_obs_in_gif', action='store_true',
                        help="If true, concatenates observation frames to the beginning of each prediction gif.")
    parser.add_argument('--shadow', action='store_true', default=False,
                        help="If true, adds shadow to the mesh visualization.")
    parser.add_argument('--x_rot', type=float, default=0,
                        help="Rotation around x-axis.")
    parser.add_argument('--y_rot', type=int, default=0,
                        help="Rotation around y-axis.")
    parser.add_argument('--z_rot', type=int, default=0,
                        help="Rotation around z-axis.")

    args = parser.parse_args()
    main(args.parent_folder, args.dest_dir, args.if_concat_obs_in_gif,
         args.x_rot, args.y_rot, args.z_rot, args.redo)
