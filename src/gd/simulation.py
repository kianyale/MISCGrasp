import argparse
import gc
from pathlib import Path
import time
import os
import numpy as np
import pybullet
import sys
from skimage import io
import tqdm.notebook

try:
    from mpi4py import MPI
except:
    pass
from tqdm import tqdm

sys.path.append("/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp")
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.gd.grasp import Label
from src.gd.perception import *
from src.gd.io import *
from src.gd.utils import btsim, workspace_lines
from src.gd.utils.transform import Rotation, Transform

from data_generator.grasp.gripper_module import load_gripper

# from memory_profiler import profile
import scipy.signal as signal


class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, gui=True, seed=None, renderer_root_dir="", args=None):
        assert scene in ["pile", "packed", "single"]

        self.urdf_root = Path(renderer_root_dir + "/data/urdfs")
        self.scene = scene
        self.object_set = object_set

        if args.gen_scene_descriptor:
            self.discover_objects(1)
        elif args.gen_test_scene_descriptor:
            self.discover_objects(2)
        elif args.load_scene_descriptor:
            self.discover_objects(3)

        self.global_scaling = {"blocks": 1.67}.get(object_set, 1.0)
        self.gui = gui

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui)
        self.gripper = None
        self.size = 0.4  # TODO 0.3
        self.table_height = 0.05
        intrinsic = CameraIntrinsic(640, 360, 459.14, 459.14, 319.75, 179.75)  # TODO
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

        self.args = args
        self.renderer_root_dir = renderer_root_dir
        if self.args.load_scene_descriptor:
            if self.scene == "pile":
                dir_name = f"pile_{object_set}_{args.obj_scale}_100"
            elif self.scene == "packed":
                dir_name = f"packed_{object_set}_{args.obj_scale}_100"
            elif self.scene == "single":
                dir_name = f"{args.gripper_type}_single_egad_{args.obj_scale}_49"
            else:
                raise NotImplementedError('No scene type!')

            scene_root_dir = os.path.join(renderer_root_dir, "data/mesh_pose_list", dir_name)
            self.scene_descriptor_list = [os.path.join(scene_root_dir, i) for i in sorted(os.listdir(scene_root_dir))]

    def change_gripper(self, gripper_type, scale):
        del self.gripper
        if gripper_type is not None and scale is not None:
            self.gripper = load_gripper(gripper_type)(self.world, scale)
        else:
            self.gripper = load_gripper('robotiq_2f_85')(self.world, 1)

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects(self, flag):
        if flag == 1:
            if self.object_set == 'pile' or self.object_set == 'packed':
                root = self.urdf_root / self.object_set / "train"
            elif self.object_set == 'egad':
                root = self.urdf_root / (self.object_set + "_train")
            elif self.object_set == 'egad_adv':
                root = self.urdf_root / self.object_set

        elif flag == 2:
            if self.object_set == 'pile' or self.object_set == 'packed':
                root = self.urdf_root / self.object_set / "test"
            elif self.object_set == 'egad':
                root = self.urdf_root / (self.object_set + "_test")
            elif self.object_set == 'egad_adv':
                root = self.urdf_root / self.object_set

        elif flag == 3:
            if self.object_set == 'pile' or self.object_set == 'packed':
                # root = self.urdf_root / self.object_set / "test"
                root = self.urdf_root / self.object_set / "train"
            elif self.object_set == 'egad':
                root = self.urdf_root / (self.object_set + "_test")
            elif self.object_set == 'egad_adv':
                root = self.urdf_root / self.object_set

        self.object_urdfs = sorted([f for f in root.iterdir() if f.suffix == ".urdf"], key=lambda p: p.stem)

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count, n_round):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=0.0,
                cameraPitch=-65,
                cameraTargetPosition=[0.2, 0.2, -0.3],
                # cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        self.place_table(self.table_height)

        if self.args.gen_scene_descriptor:
            if self.scene == "pile":
                urdfs_and_poses_dict, urdfs_and_poses_rest_list = self.generate_pile_scene(object_count,
                                                                                           self.table_height)
                return urdfs_and_poses_dict, urdfs_and_poses_rest_list
            elif self.scene == "packed":
                urdfs_and_poses_dict, urdfs_and_poses_rest_list = self.generate_packed_scene(object_count,
                                                                                             self.table_height)
                return urdfs_and_poses_dict, urdfs_and_poses_rest_list
            elif self.scene == "single":
                urdfs_and_poses_dict, urdfs_and_poses_rest_list = self.generate_packedsingle_scene(self.table_height)
                return urdfs_and_poses_dict, urdfs_and_poses_rest_list
            else:
                raise ValueError("Invalid scene argument")
        elif self.args.gen_test_scene_descriptor:
            if self.scene == "pile":
                urdfs_and_poses_dict, urdfs_and_poses_rest_list = self.generate_pile_test_scene(object_count,
                                                                                                self.table_height,
                                                                                                self.args.object_scale)
                return urdfs_and_poses_dict, urdfs_and_poses_rest_list
            elif self.scene == "packed":
                urdfs_and_poses_dict, urdfs_and_poses_rest_list = self.generate_packed_test_scene(object_count,
                                                                                                  self.table_height,
                                                                                                  self.args.object_scale)
                return urdfs_and_poses_dict, urdfs_and_poses_rest_list
            elif self.scene == "single":
                urdfs_and_poses_dict, urdfs_and_poses_rest_list = self.generate_packedsingle_test_scene(self.table_height,
                                                                                                        self.args.object_scale,
                                                                                                        n_round
                                                                                                        )
                return urdfs_and_poses_dict, urdfs_and_poses_rest_list
            else:
                raise ValueError("Invalid scene argument")
        elif self.args.load_scene_descriptor:
            scene_descriptor_npz = self.scene_descriptor_list[n_round]

            if self.scene == "pile":
                urdfs_and_poses_dict = self.generate_pile_scene(object_count, self.table_height, scene_descriptor_npz)
            elif self.scene == "packed":
                urdfs_and_poses_dict = self.generate_packed_scene(object_count, self.table_height, scene_descriptor_npz)
            elif self.scene == "single":
                urdfs_and_poses_dict = self.generate_packedsingle_scene(self.table_height, scene_descriptor_npz)
            else:
                raise ValueError("Invalid scene argument")
            return urdfs_and_poses_dict
        else:
            raise NotImplementedError

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        # pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        pose = Transform(Rotation.identity(), [0. + self.size / 2, 0. + self.size / 2, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def generate_seen_scene(self, table_height, mesh_pose_npz):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.027, 0.027, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.73)

        # read mesh_pose_npz
        print("########## scene name: ", mesh_pose_npz)
        if self.args.check_seen_scene:
            urdfs_and_poses_dict = np.load(mesh_pose_npz, allow_pickle=True)['pc']
            urdf_path_list = list(urdfs_and_poses_dict[:, 0])
            obj_scale_list = list(urdfs_and_poses_dict[:, 1])
            obj_RT_list = list(urdfs_and_poses_dict[:, 2])

        urdfs_and_poses_dict = {}  ##
        for i in range(len(urdf_path_list)):
            urdf = os.path.join(self.renderer_root_dir, urdf_path_list[i].replace("_visual.obj", ".urdf"))
            RT = obj_RT_list[i]
            R = RT[:3, :3]
            T = RT[:3, 3]
            rotation = Rotation.from_matrix(R)
            pose = Transform(rotation, T)
            scale = obj_scale_list[i]
            body = self.world.load_urdf(urdf, pose, scale=scale)
            body.set_pose(pose=Transform(rotation, T))

        # remove box
        self.world.remove_body(box)

        removed_object = True
        while removed_object:
            removed_object, urdfs_and_poses_rest_list = self.remove_objects_outside_workspace()

        for urdf, scale, rest_pose, body_uid in urdfs_and_poses_rest_list:
            urdfs_and_poses_dict[body_uid] = [scale, rest_pose.rotation.as_quat(), rest_pose.translation, str(urdf)]

        return urdfs_and_poses_dict

    def generate_pile_scene(self, object_count, table_height, scene_descriptor_npz=None):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.027, 0.027, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.73)
        # pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])  # TODO: Using this, the box length can be derived as 0.2.
        # box = self.world.load_urdf(urdf, pose, scale=1.3)

        # NOTE: urdfs_and_poses_dict is for the testing data generation, for example：
        #       '/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets/data/mesh_pose_list/pile_pile_test_200/0000.npy'
        urdfs_and_poses_dict = {}
        if self.args.gen_scene_descriptor:
            urdf_path_list = self.rng.choice(self.object_urdfs, size=object_count)
        elif self.args.load_scene_descriptor:
            dict = np.load(scene_descriptor_npz, allow_pickle=True).item()
            obj_scale_list = [value[0] for value in dict.values()]
            obj_quat_list = [value[1] for value in dict.values()]
            obj_xy_list = [value[2] for value in dict.values()]
            if self.scene != self.object_set:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[3].replace(self.scene, self.object_set))
                                  for value in dict.values()]
            else:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[3]) for value in dict.values()]

        # drop objects
        for i in range(len(urdf_path_list)):
            # import ipdb; ipdb.set_trace()
            if self.args.gen_scene_descriptor:
                rotation = Rotation.random(random_state=self.rng)
                xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
                pose = Transform(rotation, np.r_[xy, table_height + 0.2])
                if self.num_objects == 1:  # there's already been a box.
                    scale = 1.0
                elif self.num_objects == object_count:
                    scale = 0.5
                else:
                    scale = self.rng.uniform(0.5, 1.4)
                # save info
                urdfs_and_poses_dict[i] = [scale, pose.rotation.as_quat(), xy, str(urdf_path_list[i])]  # (x, y, z, w)
            elif self.args.load_scene_descriptor:
                rotation = Rotation.from_quat(obj_quat_list[i])
                xy = obj_xy_list[i]
                if self.object_set == 'egad_adv':
                    pose = Transform(rotation, np.r_[xy, table_height + 0.35])
                else:
                    pose = Transform(rotation, np.r_[xy, table_height + 0.2])
                scale = obj_scale_list[i]
            self.world.load_urdf(urdf_path_list[i], pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        # NOTE: urdfs_and_poses_rest_list is for the training data generation, for example:
        #       '/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/datasets/data/GIGA/data_pile_train_raw/mesh_pose_list/
        #       0a0a7c00a13e4a9e8d8ebe706b36869d.npz'
        urdfs_and_poses_rest_list = self.remove_and_wait()

        if self.args.gen_scene_descriptor:
            return urdfs_and_poses_dict, urdfs_and_poses_rest_list
        else:
            for urdf, scale, rest_pose, body_uid in urdfs_and_poses_rest_list:
                rest_pose = Transform.from_matrix(rest_pose)
                urdfs_and_poses_dict[body_uid] = [scale, rest_pose.rotation.as_quat(), rest_pose.translation, str(urdf)]
            return urdfs_and_poses_dict

    def generate_packed_scene(self, object_count, table_height, scene_descriptor_npz=None):
        attempts = 0
        max_attempts = 12

        urdfs_and_poses_dict = {}
        if self.args.load_scene_descriptor:
            dict = np.load(scene_descriptor_npz, allow_pickle=True).item()
            obj_scale_list = [value[0] for value in dict.values()]
            obj_angle_list = [value[1] for value in dict.values()]
            obj_x_list = [value[2] for value in dict.values()]
            obj_y_list = [value[3] for value in dict.values()]
            if self.scene != self.object_set:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[4].replace(self.scene, self.object_set))
                                  for value in dict.values()]
            else:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[4]) for value in dict.values()]

        while self.num_objects < object_count:
            self.save_state()
            if self.args.gen_scene_descriptor:
                urdf = self.rng.choice(self.object_urdfs)
                x = self.rng.uniform(0.08, self.size - 0.08)  # TODO
                y = self.rng.uniform(0.08, self.size - 0.08)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                if self.num_objects == 0:
                    scale = 1.0
                elif self.num_objects == object_count - 1:
                    scale = 0.5
                else:
                    scale = self.rng.uniform(0.5, 1.5)
                # save info
                urdfs_and_poses_dict[attempts] = [scale, angle, x, y, str(urdf)]  # (x, y, z, w)
            elif self.args.load_scene_descriptor:
                urdf = urdf_path_list[attempts]
                angle = obj_angle_list[attempts]
                x = obj_x_list[attempts]
                y = obj_y_list[attempts]
                scale = obj_scale_list[attempts]

            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            z = 1.0
            pose = Transform(rotation, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))

            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                urdfs_and_poses_rest_list = self.remove_and_wait()
            attempts += 1

        if self.args.gen_scene_descriptor:
            return urdfs_and_poses_dict, urdfs_and_poses_rest_list
        else:
            for urdf, scale, rest_pose, body_uid in urdfs_and_poses_rest_list:
                rest_pose = Transform.from_matrix(rest_pose)
                urdfs_and_poses_dict[body_uid] = [scale, rest_pose.rotation.as_quat(), rest_pose.translation, str(urdf)]

            return urdfs_and_poses_dict

    def generate_packedsingle_scene(self, table_height, scene_descriptor_npz=None):  # TODO
        attempts = 0

        if self.args.gen_scene_descriptor:
            urdfs_and_poses_dict = {}
        elif self.args.load_scene_descriptor:
            dict = np.load(scene_descriptor_npz, allow_pickle=True).item()
            obj_scale_list = [value[0] for value in dict.values()]
            obj_angle_list = [value[1] for value in dict.values()]
            obj_x_list = [value[2] for value in dict.values()]
            obj_y_list = [value[3] for value in dict.values()]
            if self.scene != self.object_set:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[4].replace(self.scene, self.object_set))
                                  for value in dict.values()]
            else:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[4]) for value in dict.values()]

        for _ in range(1):
            self.save_state()
            if self.args.gen_scene_descriptor:
                urdf = self.rng.choice(self.object_urdfs)
                x = self.rng.uniform(0.08, self.size - 0.08)
                y = self.rng.uniform(0.08, self.size - 0.08)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                scale = self.rng.uniform(0.7, 0.9)
                # save info
                urdfs_and_poses_dict[attempts] = [scale, angle, x, y, str(urdf)]  # (x, y, z, w)
            elif self.args.load_scene_descriptor:
                urdf = urdf_path_list[attempts]
                angle = obj_angle_list[attempts]
                x = obj_x_list[attempts]
                y = obj_y_list[attempts]
                scale = obj_scale_list[attempts]

            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            z = 1.0
            pose = Transform(rotation, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))

            self.world.step()

            while self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()

                body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
                z += 0.002
                body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))

                self.world.step()

            urdfs_and_poses_rest_list = self.remove_and_wait()

        if self.args.gen_scene_descriptor:
            return urdfs_and_poses_dict, urdfs_and_poses_rest_list
        else:
            remain_obj_inws_infos = []
            for body in list(self.world.bodies.values()):
                urdf = self.world.bodies_urdfs[body.uid][0]
                scale = self.world.bodies_urdfs[body.uid][1]
                if str(urdf).split("/")[-1] != "box.urdf" and str(urdf).split("/")[-1] != "plane.urdf":
                    rest_pose = body.get_pose()
                    rest_pose_quat = rest_pose.rotation.as_quat()  # (x, y, z, w)
                    rest_pose_trans = rest_pose.translation
                    remain_obj_inws_infos.append([urdf, scale, rest_pose_quat, rest_pose_trans, str(body.uid)])
            urdfs_and_poses_dict = {}
            for urdf, scale, rest_pose_quat, rest_pose_trans, body_uid in remain_obj_inws_infos:
                urdfs_and_poses_dict[body_uid] = [scale, rest_pose_quat, rest_pose_trans, str(urdf)]
            return urdfs_and_poses_dict

    def generate_pile_test_scene(self, object_count, table_height, object_scale=None):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        # pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        # box = self.world.load_urdf(urdf, pose, scale=1.3)
        pose = Transform(Rotation.identity(), np.r_[0.027, 0.027, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.73)

        # NOTE: urdfs_and_poses_dict is for the testing data generation, for example：
        #       '/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets/data/mesh_pose_list/pile_pile_test_200/0000.npy'
        urdfs_and_poses_dict = {}
        urdf_path_list = self.rng.choice(self.object_urdfs, size=object_count)

        # drop objects
        for i in range(len(urdf_path_list)):
            # import ipdb; ipdb.set_trace()
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.35])
            if object_scale == 'small':
                scale = self.rng.uniform(0.8, 1.0)
            elif object_scale == 'median':
                scale = self.rng.uniform(0.8, 1.0)
            elif object_scale == 'large':
                scale = self.rng.uniform(0.8, 1.0)
            elif object_scale == 'all':
                scale = self.rng.uniform(0.5, 1.3)
            # save info
            urdfs_and_poses_dict[i] = [scale, pose.rotation.as_quat(), xy, str(urdf_path_list[i])]  # (x, y, z, w)

            self.world.load_urdf(urdf_path_list[i], pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)

        urdfs_and_poses_rest_list = self.remove_and_wait()

        return urdfs_and_poses_dict, urdfs_and_poses_rest_list

    def generate_packed_test_scene(self, object_count, table_height, object_scale=None):
        attempts = 0
        max_attempts = 100

        urdfs_and_poses_dict = {}

        while self.num_objects < object_count: # and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            x = self.rng.uniform(0.08, self.size - 0.08)
            y = self.rng.uniform(0.08, self.size - 0.08)
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            if object_scale == 'small':
                scale = self.rng.uniform(0.7, 0.9)
            elif object_scale == 'median':
                scale = self.rng.uniform(0.7, 1.0)
            elif object_scale == 'large':
                scale = self.rng.uniform(0.8, 1.0)
            elif object_scale == 'all':
                scale = self.rng.uniform(0.5, 1.1)
            # save info
            urdfs_and_poses_dict[attempts] = [scale, angle, x, y, str(urdf)]  # (x, y, z, w)

            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            z = 1.0
            pose = Transform(rotation, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))

            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                urdfs_and_poses_rest_list = self.remove_and_wait()
            attempts += 1

        return urdfs_and_poses_dict, urdfs_and_poses_rest_list

    def generate_packedsingle_test_scene(self, table_height, object_scale=None, n_round=None):  # TODO
        attempts = 0

        urdfs_and_poses_dict = {}

        for _ in range(1):
            self.save_state()
            urdf = self.object_urdfs[n_round]
            x = self.size / 2
            y = self.size / 2
            angle = 0
            if object_scale == 'small':
                scale = 0.5
            elif object_scale == 'medium':
                scale = 0.75
            elif object_scale == 'large':
                scale = 1.05

            # save info
            urdfs_and_poses_dict[attempts] = [scale, angle, x, y, str(urdf)]  # (x, y, z, w)

            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            z = 1
            pose = Transform(rotation, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))

            self.world.step()

            while self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
                with open("./misc/singlepack_problem_object.txt", 'a') as f:
                    f.writelines(str(urdf) + '\n')

                body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
                z += 0.002
                body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))

                self.world.step()

            urdfs_and_poses_rest_list = self.remove_and_wait()
            # print(str(urdf))
        return urdfs_and_poses_dict, urdfs_and_poses_rest_list

    def acquire_tsdf(self, n, N=None, save=False, suffix=None):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.

        If N is None, the n viewpoints are equally distributed on circular trajectory.

        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, 80)
        high_res_tsdf = TSDFVolume(self.size, 160)

        origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
        r = 2.0 * self.size

        N = N if N else n
        phi_list = 2.0 * np.pi * np.arange(n) / N
        theta_list = [np.pi / 4]

        extrinsics = []
        for theta in theta_list:
            for phi in phi_list:
                extrinsics.append(camera_on_sphere(origin, r, theta, phi))

        depth_imgs, rgb_imgs = [], []
        timing = 0.0
        for i, extrinsic in enumerate(extrinsics):
            rgb_img, depth_img = self.camera.render(extrinsic)
            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)

        if save:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
            r = 0.4
            phi = - np.pi / 2
            theta = 0.3
            extrinsic = camera_on_sphere(origin, r, theta, phi)
            rgb_img, depth_img = self.camera.render(extrinsic)
            if suffix is None:
                io.imsave(f"/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/vlz/vlz_image/VGN_vs_MISC/temp.png", rgb_img)
            else:
                io.imsave(f"/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/vlz/vlz_image/medium/{suffix}.png", rgb_img)

        return tsdf, high_res_tsdf, timing

    def execute_grasp(self, grasp, remove=True, allow_contact=False, finger_depth=None):
        # -- grasp is the target containing pose and width
        # -- flag to control whether allow collision between pre-target and target
        # -- remove whether remove the object from the scene after successful grasp
        T_world_grasp = grasp.pose
        if finger_depth is not None:
            T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -finger_depth])
        else:
            T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        # approach along z-axis of the gripper
        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        # move the gripper to pregrasp pose and detect the collision
        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            result = Label.FAILURE, 1.
            # print("0")
        else:
            # move the gripper to the target pose and detect collision
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            """
            self.set_obj_pose_again(self.mesh_pose_npz)
            """
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, 1.
                # print("1")
            else:
                self.gripper.move(0.0)  # shrink the gripper
                # lift the gripper up along z-axis of the world frame or z-axis of the gripper frame
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                # print(self.gripper.read())

                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read()
                    # print("2")
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB, isRemoveObjPerGrasp=True)
                else:
                    result = Label.FAILURE, 1.
                    # print("3")
        self.world.remove_body(self.gripper.body)

        remain_obj_inws_infos = []
        if remove:
            remain_obj_inws_infos = self.remove_and_wait()  ### wait for blender to render updated scene

        return result, remain_obj_inws_infos

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object, remain_obj_inws_infos = self.remove_objects_outside_workspace()
        return remain_obj_inws_infos

    def wait_for_objects_to_rest(self, timeout=4.0, tol=0.005):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        remain_obj_inws_infos = []  ##

        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
            else:
                urdf = self.world.bodies_urdfs[body.uid][0]
                scale = self.world.bodies_urdfs[body.uid][1]
                if str(urdf).split("/")[-1] != "box.urdf" and str(urdf).split("/")[-1] != "plane.urdf":
                    rest_pose = body.get_pose()
                    remain_obj_inws_infos.append([str(urdf), scale, rest_pose.as_matrix(), str(body.uid)])
        return removed_object, remain_obj_inws_infos  ##

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        # TODO: using original gripper needs to set here
        res = len(contacts) > 0 and gripper.read() > 0.05
        # res = len(contacts) > 0 and gripper.read() > 0.05 * gripper.max_opening_width
        return res
