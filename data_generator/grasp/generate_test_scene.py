import gc
import sys

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')

import argparse
from pathlib import Path

from mpi4py import MPI
import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm

from src.gd.grasp import Grasp, Label
from src.gd.io import *
from src.gd.perception import *
from src.gd.simulation import ClutterRemovalSim
from src.gd.utils.transform import Rotation, Transform

# from memory_profiler import profile

N_ROUND = 1  # modify when running in parallel
MAX_OBJECTS_PER_SCENE = 1
OBJECT_COUNT = 10
# MAX_VIEWPOINT_COUNT = 6
VIEWPOINT_COUNT = 6
POSITIVE_GRASPS = 0  # TODO
NEGATIVE_GRASPS = 0  # TODO
SCALE_LIST = [1.]
GRIPPER_TYPES = [
    #     'franka',
    #     'robotiq_2f_85',
    #     'robotiq_2f_140',
    #     'wsg_32',
    #     'ezgripper',
    #     'sawyer',
    #     'wsg_50',
    #     'rg2',
    #     'barrett_hand_2f',
    #     'kinova_3f',
    #     'robotiq_3f',
    #     'barrett_hand'
]
"""
ave:
    small: 0.05
    0.6
    median: 0.08655
    1.33
    large: 0.115
    # 'franka': 0.08
    # 'robotiq_2f_85': 0.085
    # 'robotiq_2f_140': 0.140
    # 'wsg_32': 0.056
    # 'ezgripper': 0.105
    # 'sawyer': 0.044
    # 'wsg_50': 0.100
    # 'rg2': 0.0812
    # 'barrett_hand_2f': 0.095
"""

def main(args):
    workers, rank = setup_mpi()

    # prepare
    pos_grasps = POSITIVE_GRASPS
    neg_grasps = NEGATIVE_GRASPS
    scale_list = SCALE_LIST
    gripper_types = GRIPPER_TYPES

    sim = ClutterRemovalSim(args.scene, args.object_set, args.sim_gui, args.seed + rank, args.renderer_root_dir, args)
    pbar1 = tqdm(total=pos_grasps, desc='POSITIVE_GRASPS_PER_SCENE', disable=rank != 0)
    pbar2 = tqdm(total=len(scale_list), desc='SCALES', disable=rank != 0)
    pbar3 = tqdm(total=len(gripper_types), desc='GRIPPER_TYPES', disable=rank != 0)
    pbar4 = tqdm(total=N_ROUND, desc='N_ROUND', disable=rank != 0)

    info_path = args.root / "mesh_pose_list" / f'{args.scene}_{args.object_set}_{args.object_scale}_{args.rounds}'
    scene_path = args.root / "sensor_data" / f'{args.scene}_{args.object_set}_{args.object_scale}_{args.rounds}'
    grasp_path = args.root / "grasps" / f'{args.scene}_{args.object_set}_{args.object_scale}_{args.rounds}'

    if rank == 0:
        scene_path.mkdir(parents=True, exist_ok=True)
        info_path.mkdir(parents=True, exist_ok=True)
        grasp_path.mkdir(parents=True, exist_ok=True)
        write_setup(
            args.root,
            sim.size,
            sim.camera.intrinsic
        )

    for i in range(N_ROUND):
        # generate heap
        object_count = OBJECT_COUNT
        urdfs_and_poses_dict, urdfs_and_poses_rest_list = sim.reset(object_count, args.seed)
        sim.save_state()

        # render synthetic depth images
        n = VIEWPOINT_COUNT
        depth_imgs, extrinsics = render_images(sim, n)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(sim.size, 160, depth_imgs, sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud()
        # o3d.visualization.draw_geometries([pc])

        # crop surface and borders from point cloud
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue

        # store the raw data
        scene_file = scene_path / f'{args.seed:04d}.npz'
        np.savez_compressed(scene_file, depth_imgs=depth_imgs, extrinsics=extrinsics)
        info_file = info_path / (f'{args.seed:04d}' + ".npy")  # TODO
        np.save(info_file, urdfs_and_poses_dict)

        for gripper_type in gripper_types:
            for scale in scale_list:
                sim.change_gripper(gripper_type, scale)
                finger_depth = sim.gripper.finger_depth
                pre_depth = sim.gripper.finger_depth_init if hasattr(sim.gripper, 'finger_depth_init') else sim.gripper.finger_depth

                positive_cnt = 0
                negative_cnt = 0
                while positive_cnt < pos_grasps or negative_cnt < neg_grasps:
                    point, normal = sample_grasp_point(pc, finger_depth)
                    grasp, label = evaluate_grasp_point(sim, point, normal, pre_depth)

                    # store the sample
                    write_grasp(grasp_path, f'{args.seed:04d}', gripper_type, scale, grasp, label)
                    if label == Label.SUCCESS:
                        positive_cnt += 1
                        pbar1.update()
                    gc.collect()
                pbar1.reset()
                pbar2.update()
            gc.collect()
            pbar2.reset()
            pbar3.update()
        pbar3.reset()
        pbar4.update()

    pbar1.close()
    pbar2.close()
    pbar3.close()
    pbar4.close()


def setup_mpi():
    workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return workers, rank


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)  # TODO: n
    depth_imgs = np.empty((n, height, width), np.float32)

    phi_list = 2.0 * np.pi * np.arange(n) / n
    theta = np.random.uniform(np.pi / 6.0, np.pi / 3.0)

    cnt = 0
    for phi in phi_list:
        r = np.random.uniform(1.25, 2.0) * sim.size  # if size == 0.4: 0.5 ~ 0.8

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[cnt] = extrinsic.to_list()
        depth_imgs[cnt] = depth_img

        cnt += 1

    return depth_imgs, extrinsics


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is pointing upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, pre_depth, num_rotations=12):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, 2 * np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=1.)
        (outcome, width), _ = sim.execute_grasp(candidate, remove=False, finger_depth=pre_depth)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0],  # fill in 0 before and after the array
            height=1,
            width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1  # so here needs to minus 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed", "single"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--sim-gui", action="store_true")

    parser.add_argument("--gen-scene-descriptor", type=bool, default=False)
    parser.add_argument("--load-scene-descriptor", type=bool, default=False)
    parser.add_argument("--gen-test-scene-descriptor", type=bool, default=True)
    parser.add_argument("--object-scale", type=str, choices=['small', 'medium', 'large', 'all'],
                        default='medium')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--renderer-root-dir", type=str, default="/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets")

    args = parser.parse_args()
    main(args)
