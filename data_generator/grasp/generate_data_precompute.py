import gc
import sys

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')

import argparse
from pathlib import Path

from mpi4py import MPI
import numpy as np
import open3d as o3d
import scipy.signal as signal
from scipy.ndimage import binary_dilation
from scipy.interpolate import interpn
from tqdm import tqdm

from src.gd.grasp import Grasp, Label
from src.gd.io import *
from src.gd.perception import *
from src.gd.simulation import ClutterRemovalSim
from src.gd.utils.transform import Rotation, Transform

# from memory_profiler import profile

N_ROUND = 1  # modify when running in parallel
MAX_OBJECTS_PER_SCENE = 12
OBJECT_COUNT_LAMBDA = 4
# MAX_VIEWPOINT_COUNT = 6
VIEWPOINT_COUNT = 6
POSITIVE_GRASPS = 38  # TODO 32 + 6
NEGATIVE_GRASPS = 80  # TODO 64 + 16
# SCALE_LIST = [0.7, 0.85, 1.0, 1.15, 1.3]

SCALE_LIST = [1.]
GRIPPER_TYPES = [
    # 'franka',
    # 'robotiq_2f_85',
    # 'robotiq_2f_140',
    # 'wsg_32',
    # 'ezgripper',
    # 'sawyer',
    # 'wsg_50',
    # 'rg2',
    # 'barrett_hand_2f',
    # 'kinova_3f',
    # 'robotiq_3f',
    'barrett_hand'
]
two_finger = [
    'franka',
    'robotiq_2f_85',
    'robotiq_2f_140',
    'wsg_32',
    'ezgripper',
    'sawyer',
    'wsg_50',
    'rg2',
    'barrett_hand_2f',
]
three_finger = [
    'kinova_3f',
    'robotiq_3f',
    'barrett_hand'
]
log_file = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/time_ave_official.txt')
structure = np.ones([1, 3, 3, 3], dtype=bool)
structure[:, :, :, 0:1] = False


def main(args):
    workers, rank = setup_mpi()

    # prepare
    # pos_grasps = np.random.choice(np.arange(POSITIVE_GRASPS - 5, POSITIVE_GRASPS + 6))
    pos_grasps = POSITIVE_GRASPS
    neg_grasps = NEGATIVE_GRASPS
    if args.scale_augmentation:
        foo = np.random.choice([0, 1])
        if foo == 1:
            scale_list = np.random.choice(SCALE_LIST, 3, replace=False, p=[0.125, 0.25, 0.25, 0.25, 0.125])
            gripper_types = np.random.choice(GRIPPER_TYPES, 1, replace=False)
        else:
            scale_list = [1]
            gripper_types = np.random.choice(GRIPPER_TYPES, 3, replace=False)
    else:
        scale_list = [1]
        gripper_types = GRIPPER_TYPES

    sim = ClutterRemovalSim(args.scene, args.object_set, args.sim_gui, args.seed + rank, args.renderer_root_dir, args)
    pbar1 = tqdm(total=pos_grasps, desc='POSITIVE_GRASPS_PER_SCENE', disable=rank != 0)
    pbar2 = tqdm(total=len(scale_list), desc='SCALES', disable=rank != 0)
    pbar3 = tqdm(total=len(gripper_types), desc='GRIPPER_TYPES', disable=rank != 0)
    pbar4 = tqdm(total=N_ROUND, desc='N_ROUND', disable=rank != 0)

    if rank == 0:
        (args.root / "sensor_data").mkdir(parents=True, exist_ok=True)
        (args.root / "mesh_pose_list").mkdir(parents=True, exist_ok=True)
        write_setup(
            args.root,
            sim.size,
            sim.camera.intrinsic
        )

    for i in range(N_ROUND):
        # generate heap
        object_count = min(np.random.poisson(OBJECT_COUNT_LAMBDA) + 1, MAX_OBJECTS_PER_SCENE)
        object_count = max(2, object_count)
        _, urdfs_and_poses_rest_list = sim.reset(object_count, i)
        sim.save_state()

        # render synthetic depth images
        n = VIEWPOINT_COUNT
        depth_imgs, extrinsics = render_images(sim, n)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(sim.size, 160, depth_imgs, sim.camera.intrinsic, extrinsics)
        grid = tsdf.get_grid()
        reg_grid = regularize_tsdf_grid(grid)
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
        scene_id = write_sensor_data(args.root, depth_imgs, extrinsics)
        path = args.root / "mesh_pose_list" / (scene_id + ".npz")
        np.savez_compressed(path, pc=np.asarray(urdfs_and_poses_rest_list, dtype=object))  # NOTE: structured array

        for gripper_type in gripper_types:
            start_time = time.time()
            for scale in scale_list:
                sim.change_gripper(gripper_type, scale)
                finger_depth = sim.gripper.finger_depth
                positive_cnt = 0
                negative_cnt = 0
                while positive_cnt < pos_grasps or negative_cnt < neg_grasps:
                    results = sample_evaluate_grasp_point(sim, pc, reg_grid, sim.gripper)
                    for grasp, label in results:
                        # store the sample
                        write_grasp(args.root, scene_id, gripper_type, scale, grasp, int(label))
                        if label == Label.SUCCESS:
                            positive_cnt += 1
                            pbar1.update()
                        else:
                            negative_cnt += 1
                    gc.collect()
                pbar1.reset()
                pbar2.update()

            elapsed_time = time.time() - start_time  # 计算循环时间
            log_gripper_time(gripper_type, elapsed_time, log_file)  # 记录时间到txt文件

            gc.collect()
            pbar2.reset()
            pbar3.update()
        pbar3.reset()
        pbar4.update()

    pbar1.close()
    pbar2.close()
    pbar3.close()
    pbar4.close()


def regularize_tsdf_grid(tsdf_grid, threshold=0.9):
    """
    对 TSDF 网格中的体素进行扩张操作，直到外部截断的体素都变为1。

    Args:
        tsdf_grid (numpy.ndarray): 形状为 (1, res_x, res_y, res_z) 的 TSDF 网格
        threshold (float): 值大于该阈值的体素将作为扩张种子

    Returns:
        numpy.ndarray: 扩张后的 TSDF 网格
    """
    # 获取 TSDF grid 的形状
    grid_shape = tsdf_grid.shape

    # 找到值大于阈值（0.9）的体素，作为种子区域
    seed_mask = tsdf_grid > threshold
    protect_mask = tsdf_grid == 0.

    # 将膨胀区域的值设为 True，截断的体素 (值为 0) 将被扩展到 1
    dilated_mask = seed_mask.copy()  # 初始化膨胀区域
    new_dilated_mask = binary_dilation(dilated_mask, structure=structure, iterations=0, mask=protect_mask)

    tsdf_grid[new_dilated_mask & (tsdf_grid == 0)] = 1.

    return tsdf_grid


def trilinear_interpolation_scipy(tsdf_grid, points, voxel_size=0.0025, grid_origin=np.zeros([1, 3])):
    """
    使用 scipy.interpolate.interpn 实现三线性插值

    :param tsdf_grid: 形状为 (1, i, j, k) 的 TSDF 网格
    :param point: 任意3D世界坐标 (n, 3)
    :param voxel_size: 每个体素的大小
    :param grid_origin: TSDF 网格的原点坐标 (1, 3)
    :return: 指定点的 TSDF 值
    """
    # 生成网格坐标
    i_range = np.arange(tsdf_grid.shape[1])
    j_range = np.arange(tsdf_grid.shape[2])
    k_range = np.arange(tsdf_grid.shape[3])
    grid_coords = (i_range, j_range, k_range)

    # 计算点在体素网格中的相对坐标
    local_points = (points - grid_origin) / voxel_size

    # 使用 interpn 进行三线性插值,[n,]
    tsdf_value = interpn(grid_coords, tsdf_grid[0], local_points, method='linear', bounds_error=False, fill_value=None)

    return tsdf_value


def check_gripper_collision(point, normal, grid, gripper, num_rotations=12):
    potential_poses = []

    if gripper.name in two_finger:
        depth = gripper.finger_depth_init if hasattr(gripper, 'finger_depth_init') else gripper.finger_depth
        openning = gripper.max_opening_width
        flag = 2
    elif gripper.name in three_finger:
        depth = gripper.finger_depth_init if hasattr(gripper, 'finger_depth_init') else gripper.finger_depth
        left_openning = gripper.finger_open_distance_left
        right_openning = gripper.finger_open_distance_right
        half_gap = gripper.half_gap_2f
        flag = 3

    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):  # avoid collinearity
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, 2 * np.pi, num_rotations)

    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        pose = Transform(ori, point)
        pose_pre = Transform(ori, point) * Transform(Rotation.identity(), [0.0, 0.0, -depth])

        key_pts = [point, pose_pre.translation]

        if flag == 2:
            left_finger_end = (pose * Transform(Rotation.identity(),
                                                [0.0, -openning / 2, depth])).translation
            key_pts.append(left_finger_end)
            right_finger_end = (pose * Transform(Rotation.identity(),
                                                 [0.0, openning / 2, depth])).translation
            key_pts.append(right_finger_end)
            left_finger_start = (pose * Transform(Rotation.identity(),
                                                  [0.0, -openning / 2, 0.0])).translation
            key_pts.append(left_finger_start)
            right_finger_start = (pose * Transform(Rotation.identity(),
                                                   [0.0, openning / 2, 0.0])).translation
            key_pts.append(right_finger_start)

            pre_left_finger_end = (pose_pre * Transform(Rotation.identity(),
                                                        [0.0, -openning / 2, depth])).translation
            key_pts.append(pre_left_finger_end)
            pre_right_finger_end = (pose_pre * Transform(Rotation.identity(),
                                                         [0.0, openning / 2, depth])).translation
            key_pts.append(pre_right_finger_end)
            pre_left_finger_start = (pose_pre * Transform(Rotation.identity(),
                                                          [0.0, -openning / 2, 0.0])).translation
            key_pts.append(pre_left_finger_start)
            pre_right_finger_start = (pose_pre * Transform(Rotation.identity(),
                                                           [0.0, openning / 2, 0.0])).translation
            key_pts.append(pre_right_finger_start)

        elif flag == 3:
            right_finger_end = (pose * Transform(Rotation.identity(),
                                                [0.0, right_openning, depth])).translation
            key_pts.append(right_finger_end)
            left_finger_end_front = (pose * Transform(Rotation.identity(),
                                                       [half_gap, -left_openning, depth])).translation
            key_pts.append(left_finger_end_front)
            left_finger_end_back = (pose * Transform(Rotation.identity(),
                                                      [-half_gap, -left_openning, depth])).translation
            key_pts.append(left_finger_end_back)
            right_finger_start = (pose * Transform(Rotation.identity(),
                                                [0.0, right_openning, 0.0])).translation
            key_pts.append(right_finger_start)
            left_finger_start_front = (pose * Transform(Rotation.identity(),
                                                       [half_gap, -left_openning, 0.0])).translation
            key_pts.append(left_finger_start_front)
            left_finger_start_back = (pose * Transform(Rotation.identity(),
                                                      [-half_gap, -left_openning, 0.0])).translation
            key_pts.append(left_finger_start_back)

            pre_right_finger_end = (pose_pre * Transform(Rotation.identity(),
                                                [0.0, right_openning, depth])).translation
            key_pts.append(pre_right_finger_end)
            pre_left_finger_end_front = (pose_pre * Transform(Rotation.identity(),
                                                       [half_gap, -left_openning, depth])).translation
            key_pts.append(pre_left_finger_end_front)
            pre_left_finger_end_back = (pose_pre * Transform(Rotation.identity(),
                                                      [-half_gap, -left_openning, depth])).translation
            key_pts.append(pre_left_finger_end_back)
            pre_right_finger_start = (pose_pre * Transform(Rotation.identity(),
                                                [0.0, right_openning, 0.0])).translation
            key_pts.append(pre_right_finger_start)
            pre_left_finger_start_front = (pose_pre * Transform(Rotation.identity(),
                                                       [half_gap, -left_openning, 0.0])).translation
            key_pts.append(pre_left_finger_start_front)
            pre_left_finger_start_back = (pose_pre * Transform(Rotation.identity(),
                                                      [-half_gap, -left_openning, 0.0])).translation
            key_pts.append(pre_left_finger_start_back)

        key_pts = np.asarray(key_pts)
        tsdf_val = trilinear_interpolation_scipy(grid, key_pts)

        if np.all(tsdf_val > 0.5):
            potential_poses.append(pose)

    return potential_poses


def log_gripper_time(gripper_type, elapsed_time, log_file):
    # 记录每次gripper_type的时间到txt文件中
    with open(log_file, 'a') as f:
        f.write(f"{gripper_type}: {elapsed_time:.2f} seconds\n")


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
        # r = np.random.uniform(1.6, 2.4) * sim.size  # if size == 0.3: 0.48 ~ 0.72
        r = np.random.uniform(1.25, 2.0) * sim.size  # if size == 0.4: 0.5 ~ 0.8

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[cnt] = extrinsic.to_list()
        depth_imgs[cnt] = depth_img

        cnt += 1

    return depth_imgs, extrinsics


def sample_evaluate_grasp_point(sim, point_cloud, grid, gripper, eps=0.1):
    finger_depth = gripper.finger_depth
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]

        # pointing upwards
        grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
        point = point + normal * grasp_depth

        ok = normal[2] > -0.1
        if ok:
            potential_poses = check_gripper_collision(point, normal, grid, gripper)  # make sure the normal is pointing upwards
            if potential_poses:
                ok = True
            else:
                ok = False

    pre_depth = gripper.finger_depth_init if hasattr(gripper, 'finger_depth_init') else gripper.finger_depth

    results = []
    for pose in potential_poses:
        sim.restore_state()
        candidate = Grasp(pose, width=1.)
        (outcome, width), _ = sim.execute_grasp(candidate, remove=False, finger_depth=pre_depth)
        if outcome == Label.SUCCESS:
            results.append((Grasp(pose, width=width), outcome))

    if not results:
        results.append((Grasp(np.random.choice(potential_poses), width=1.), Label.FAILURE))
    return results


def evaluate_grasp_point(sim, pos, normal, num_rotations=6, finger_depth=None):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):  # avoid collinearity
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=1.)
        (outcome, width), _ = sim.execute_grasp(candidate, remove=False, finger_depth=finger_depth)
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

    parser.add_argument("--gen-scene-descriptor", type=bool, default=True)
    parser.add_argument("--gen_test_scene_descriptor", type=bool, default=False)
    parser.add_argument("--load-scene-descriptor", type=bool, default=False)
    parser.add_argument("--scale-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--renderer-root-dir", type=str, default="/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets")

    args = parser.parse_args()
    main(args)
