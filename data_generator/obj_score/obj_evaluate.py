import gc
import sys
import time
import warnings

import trimesh
import yaml

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')

import argparse
from pathlib import Path

from mpi4py import MPI
import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm, TqdmWarning
from colorama import *

from src.gd.grasp import Grasp, Label
from src.gd.io import *
from src.gd.perception import *
from src.gd.simulation import ClutterRemovalSim
from src.gd.utils.transform import Rotation, Transform

# from memory_profiler import profile

N_ROUND = 1  # modify when running in parallel
MAX_OBJECTS_PER_SCENE = 1
OBJECT_COUNT = 1
# MAX_VIEWPOINT_COUNT = 6
VIEWPOINT_COUNT = 6
POSITIVE_GRASPS = 1  # TODO
SCALE_LIST = [1.]
GRIPPER_TYPES = [
    'franka',
    'robotiq_2f_85',
    'robotiq_2f_140',
    'wsg_32',
    'ezgripper',
    'sawyer',
    'wsg_50',
    'rg2',
    'barrett_hand_2f',
    'kinova_3f',
    'robotiq_3f',
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
"""
ave:
    small: 0.05
    0.6
    median: 0.08655
    1.33
    large: 0.115
    
    # 'franka': 0.08    !
    # 'rg2': 0.0812
    # 'robotiq_2f_85': 0.085
    # 'wsg_50': 0.100
    # 'ezgripper': 0.112
    # 'kinova_3f': 0.084
    
    
    # 'sawyer': 0.044
    # 'wsg_32': 0.056
    # 'h5_hand': 0.07   !
    
    # 'barrett_hand_2f': 0.14
    # 'barrett_hand: 0.124  !
    # 'robotiq_3f': 0.16
    # 'robotiq_2f_140': 0.140
    
    # 'leap_hand_right': TBD
    
"""
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
    theta = np.pi / 4
    r = 1.25 * sim.size  # 0.5

    cnt = 0
    for phi in phi_list:
        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[cnt] = extrinsic.to_list()
        depth_imgs[cnt] = depth_img

        cnt += 1

    return depth_imgs, extrinsics


def sample_grasp_point(point_cloud, finger_depth, eps=0.):
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
    succ, fail = [], []
    pose = None
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        pose = Transform(ori, pos)
        sim.restore_state()
        candidate = Grasp(pose, width=1.)
        (outcome, width), _ = sim.execute_grasp(candidate, remove=False, finger_depth=pre_depth)
        if outcome == Label.SUCCESS:
            succ.append((Grasp(pose, width=width), outcome))
        else:
            fail.append((Grasp(pose, width=1.), outcome))

    return succ, fail


def fusion_func(x, y, w_x=0.7, w_y=0.3):
    return w_x * x + w_y * y


class ObjEvaluator(object):
    def __init__(self, root, renderer_root_dir, obj_set, obj_scale, interval, seed, sim_gui, args):
        self.args = args
        self.root = root
        self.info_path = root / "mesh_pose_list" / f'{args.gripper}_{args.scene}_{obj_set}_{obj_scale}_{args.rounds}'
        self.scene_path = root / "sensor_data" / f'{args.gripper}_{args.scene}_{obj_set}_{obj_scale}_{args.rounds}'
        self.grasp_path = root / "grasps" / f'{args.gripper}_{args.scene}_{obj_set}_{obj_scale}_{args.rounds}'

        self.obj_scale = obj_scale
        self.interval = interval * 60  # unit: s

        self.sim = ClutterRemovalSim('single', obj_set, sim_gui, seed, renderer_root_dir, args)
        self.urdf_root = self.sim.urdf_root
        self.urdf_path_list = self.sim.object_urdfs
        if obj_set == 'egad':
            self.obj_path_list, self.obj_stem_list = self.find_obj_files(self.urdf_root / (obj_set + '_test'))
        else:
            self.obj_path_list, self.obj_stem_list = self.find_obj_files(self.urdf_root / obj_set / 'test')

        self.gripper = args.gripper

        self.scene_path.mkdir(parents=True, exist_ok=True)
        self.info_path.mkdir(parents=True, exist_ok=True)
        self.grasp_path.mkdir(parents=True, exist_ok=True)
        write_setup(
            args.root,
            self.sim.size,
            self.sim.camera.intrinsic
        )

    @staticmethod
    def find_obj_files(dir):
        generator = dir.rglob('*.obj')  # NOTE: In Python, a generator is a one-time iterable object. Once iteration is
                                        #       complete, the generator is "exhausted" and cannot be reused.

        path_list = [path for path in generator if
                     not (str(path).endswith('_vhacd.obj') or str(path).endswith('_collision.obj'))]
        path_list = sorted(path_list, key=lambda p: p.stem)  # 按文件名排序
        stem_list = [path.stem for path in path_list if
                     not (str(path).endswith('_vhacd.obj') or str(path).endswith('_collision.obj'))]
        name_list = sorted(stem_list)
        return path_list, stem_list

    @staticmethod
    def compute_complexity(obj_path):
        # print(f"Starting geometry complexity computation for {Fore.GREEN}{obj_path.stem}{Fore.RESET}:")

        mesh = trimesh.load(obj_path)
        flat_vertices = mesh.faces.flatten()
        flat_angles = mesh.face_angles.flatten()
        unique_vertices = np.unique(flat_vertices)
        vertex_angles = np.array([2 * np.pi - flat_angles[flat_vertices == v_id].sum() for v_id in unique_vertices])

        # Create Normalised Histogram
        hist = np.histogram(vertex_angles, bins=512, range=(-np.pi * 2, np.pi * 2))[0].astype(np.float64)
        hist /= hist.sum()

        # Compute entropy
        H = -1 * (hist * np.log2(hist + 1e-6)).sum()
        H = max(0, H)

        # print(f"Computing on {Fore.GREEN}{obj_path.stem}{Fore.RESET} is done. Complexity: {Fore.GREEN}{H / 9}{Fore.RESET}")

        return H

    def compute_difficulty(self, n):
        urdfs_and_poses_dict, urdfs_and_poses_rest_list = self.sim.reset(1, n)
        self.sim.save_state()

        num_view = VIEWPOINT_COUNT
        depth_imgs, extrinsics = render_images(self.sim, num_view)

        # reconstrct point cloud using a subset of the images
        tsdf = create_tsdf(self.sim.size, 160, depth_imgs, self.sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud()
        # o3d.visualization.draw_geometries([pc])

        # crop surface and borders from point cloud
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.sim.lower, self.sim.upper)
        pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            raise NotImplementedError("Point cloud empty, interrupt")

        scene_file = self.scene_path / f'{self.obj_stem_list[n]}'
        np.savez_compressed(scene_file, depth_imgs=depth_imgs, extrinsics=extrinsics)
        info_file = self.info_path / (f'{self.obj_stem_list[n]}' + ".npy")  # TODO
        np.save(info_file, urdfs_and_poses_dict)

        res = {}
        print(f"Starting grasp difficulty computation for {Fore.GREEN}{self.gripper}{Fore.RESET}:")

        self.sim.change_gripper(self.gripper, 1)
        finger_depth = self.sim.gripper.finger_depth
        pre_depth = self.sim.gripper.finger_depth_init \
            if hasattr(self.sim.gripper, 'finger_depth_init') else self.sim.gripper.finger_depth

        positive_cnt = 0
        negative_cnt = 0
        start = time.time()
        with tqdm(total=self.interval, desc="Processing", dynamic_ncols=True, unit="s") as pbar:
            while time.time() - start < self.interval:
                point, normal = sample_grasp_point(pc, finger_depth)
                succ, fail = evaluate_grasp_point(self.sim, point, normal, pre_depth)

                for grasp, label in succ:
                    write_grasp(self.grasp_path, self.obj_stem_list[n], self.gripper, 1.0, grasp, int(label))
                    positive_cnt += 1

                for grasp, label in fail:
                    write_grasp(self.root, self.obj_stem_list[n], self.gripper, 1.0, grasp, int(label))
                    negative_cnt += 1

                elapsed_time = time.time() - start
                pbar.set_postfix(positive=positive_cnt, negative=negative_cnt, refresh=True)
                pbar.update(elapsed_time - pbar.n)  # unit conversion

                gc.collect()

            difficulty = 1. - positive_cnt / (positive_cnt + negative_cnt)
            res['difficulty'] = difficulty
            res['num_positive'] = positive_cnt
            res['num_negative'] = negative_cnt

        print(f"Computing on {Fore.GREEN}{self.gripper}{Fore.RESET} is done. Difficulty: {Fore.GREEN}{difficulty}{Fore.RESET}")

        return res

    def urdf_path2obj_path(self, urdf_path):
        if urdf_path.stem in self.obj_stem_list:
            return urdf_path.parent / (urdf_path.stem + '.obj')
        elif urdf_path.stem + '_visual' in self.obj_stem_list:
            return urdf_path.parent / (urdf_path.stem + '_visual.obj')
        else:
            for path in (urdf_path.parent / urdf_path.stem).rglob('*.obj'):
                if not str(path).endswith('_vhacd.obj') or str(path).endswith('_collision.obj'):
                    return path

    def run(self):
        res = {}
        for i in range(0, len(self.urdf_path_list)):
            stem = self.urdf_path_list[i].stem
            print(f'\nProcess on ' + Fore.GREEN + f'{stem}' + Fore.RESET + ' starts:')
            obj_path = self.urdf_path2obj_path(self.urdf_path_list[i])
            shape_complexity = ObjEvaluator.compute_complexity(obj_path)
            grasp_difficulty = self.compute_difficulty(i)  # dict
            # score = fusion_func(grasp_difficulty, shape_complexity)
            res[stem] = float(shape_complexity / 9), grasp_difficulty  # value of complexity ranges from 0 to 9
            print(f'Process on ' + Fore.GREEN + f'{stem}' + Fore.RESET + f' ends: Complexity {Fore.GREEN}{shape_complexity / 9}{Fore.RESET}.\n')

        return res

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=TqdmWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed", "single"], default="single")
    parser.add_argument("--object-set", type=str, default="egad")
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--gripper", type=str, choices=[
        'franka',
        'robotiq_2f_85',
        'robotiq_2f_140',
        'wsg_32',
        'ezgripper',
        'sawyer',
        'wsg_50',
        'rg2',
        'barrett_hand_2f',
        'kinova_3f',
        'robotiq_3f',
        'barrett_hand',
        # 'leap_hand_right',
        'h5_hand'
    ])

    parser.add_argument("--gen-scene-descriptor", type=bool, default=False)
    parser.add_argument("--load-scene-descriptor", type=bool, default=False)
    parser.add_argument("--gen-test-scene-descriptor", type=bool, default=True)
    parser.add_argument("--object-scale", type=str, choices=['small', 'medium', 'large', 'None'],
                        default='medium')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=float, default=5)  # unit: m
    parser.add_argument("--rounds", type=int, default=49)
    parser.add_argument("--renderer-root-dir", type=str, default="/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets")

    args = parser.parse_args()

    obj_eval = ObjEvaluator(args.root,
                            args.renderer_root_dir,
                            args.object_set,
                            args.object_scale,
                            args.interval,
                            args.seed,
                            args.sim_gui,
                            args)
    res = obj_eval.run()

    with open(f'./score_raw/{args.gripper}_{args.object_scale}_score_raw.yaml', 'w') as f:
        yaml.dump(res, f)

    del obj_eval.sim

    # raise
    # alphabet = [chr(i) for i in range(ord('A'), ord('G') + 1)]
    # for i in alphabet:
    #     for j in range(7):
    #         path3 = Path(f'/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/egad/egadevalset/egad_eval_set/{i}{j}.obj')
    #         H3 = ObjEvaluator.compute_complexity(path3)
    #         path4 = Path(f'/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/object2urdf/ws/check/{i}{j}.obj')
    #         H4 = ObjEvaluator.compute_complexity(path4)
    #         print(f'{i}{j}: ', H3, ' ', H4)

