import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from torch.utils.data import Dataset

from src.miscgrasp.asset import *


class PinchGraspDataset(Dataset):
    default_cfg = {}

    def __init__(self, cfg, is_train, scn_augment=False, grp_augment=False):
        self.cfg = {**self.default_cfg, **cfg}
        self.is_train = is_train
        if self.is_train:
            self.scenes = vgn_train_scene_names
            self.scn_augment = scn_augment
            self.grp_augment = grp_augment
        else:
            self.scenes = vgn_val_scene_names #[self.cfg['val_database_name']]
            self.scn_augment = self.grp_augment = False
        self.df_pile = VGN_PILE_TRAIN_CSV
        self.df_packed = VGN_PACK_TRAIN_CSV

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        outputs = {'geo_info': {},
                   'grasp_info': {}}

        split = self.scenes[i].split('/')
        grp, scene_id, scene_type = split[4], split[3], split[2]

        if scene_type == 'pile':
            df = self.df_pile[self.df_pile['scene_id'] == scene_id]
            voxel_grid = read_voxel_grid(VGN_TRAIN_ROOT / scene_type, scene_id)
        elif scene_type == 'packed':
            df = self.df_packed[self.df_packed['scene_id'] == scene_id]
            voxel_grid = read_voxel_grid(VGN_TRAIN_ROOT / scene_type, scene_id)
        else:
            raise NotImplementedError(f"No scene_type {scene_type}. Please check again!")

        df_grip_scl = df[(df['gripper_type'] == grp)]
        if self.is_train:
            df_grip_scl_pos = df_grip_scl[df_grip_scl['label'] == 1].sample(n=self.cfg['pos_num'])
            df_grip_scl_neg = df_grip_scl[df_grip_scl['label'] == 0].sample(n=self.cfg['neg_num'])
            df_grip_scl = pd.concat([df_grip_scl_pos, df_grip_scl_neg], axis=0, ignore_index=True)
        else:
            df_grip_scl_pos = df_grip_scl[df_grip_scl['label'] == 1].iloc[:self.cfg['pos_num']]
            df_grip_scl_neg = df_grip_scl[df_grip_scl['label'] == 0].iloc[:self.cfg['neg_num']]
            df_grip_scl = pd.concat([df_grip_scl_pos, df_grip_scl_neg], axis=0, ignore_index=True)

        pos = df_grip_scl.loc[:, 'i':'k'].to_numpy(np.single)
        ori = Rotation.from_quat(df_grip_scl.loc[:, "qx":"qw"].to_numpy(np.single))
        width = df_grip_scl.loc[:, "width"].to_numpy(np.single)  # TODO
        label = df_grip_scl.loc[:, "label"].to_numpy(np.single)

        if self.scn_augment:
            voxel_grid, ori, pos = apply_transform_scene(voxel_grid, ori, pos, scene_type)

        index = np.clip(np.round(pos), 0, self.cfg['volume_res'] - 1).astype(np.int64)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotation1 = ori.as_quat().astype(np.single)
        rotation2 = (ori * R).as_quat().astype(np.single)
        rotation = np.stack([rotation1, rotation2], axis=1)
        pos = pos.astype(np.single)

        outputs['grasp_info']['pos'] = pos  # (24, 3)
        outputs['grasp_info']['index'] = index
        outputs['grasp_info']['label'] = label
        outputs['grasp_info']['rotation'] = rotation  # (24, 2, 4)
        outputs['grasp_info']['width'] = width * 16

        outputs['geo_info']['scene'] = voxel_grid * 2. - 1.

        return outputs


def apply_transform_scene(voxel_grid, orientation, position, scene_type):
    angle = np.pi / 2.0 * np.random.choice(4)
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    h_max = np.max(position, axis=0)[-1]

    if scene_type == 'packed':
        z_offset = np.random.uniform(-9, 80 - h_max - 2)  # NOTE
    elif scene_type == 'pile':
        z_offset = np.random.uniform(-9, 80 - h_max - 2)
    else:
        raise NotImplementedError(f"No scene_type {scene_type}. Please check again!")

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[40.0, 40.0, 40.0])  # NOTE
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    orientation = T.rotation * orientation

    return voxel_grid, orientation, position
