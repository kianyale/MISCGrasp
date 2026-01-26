from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')
from src.gd.utils.transform import Transform, Rotation


def augment_grasps(group, aug_type):
    augmented_group = group.copy()

    positions = augmented_group.loc[:, 'x': 'z'].to_numpy(np.double)
    orientations = Rotation.from_quat(augmented_group.loc[:, "qx":"qw"].to_numpy(np.double))

    angle = np.pi / 2.0 * aug_type
    T_aug = Transform(Rotation.from_rotvec(np.r_[0., 0., angle]), [0., 0., 0.])
    T_center = Transform(Rotation.identity(), np.r_[0.2, 0.2, 0.2])
    T = T_center * T_aug * T_center.inverse()

    positions = T.transform_point(positions)
    orientations = T.rotation * orientations

    augmented_group[['x', 'y', 'z']] = positions
    augmented_group[['qx', 'qy', 'qz', 'qw']] = orientations.as_quat()
    augmented_group['aug'] = aug_type

    return augmented_group


if __name__ == "__main__":
    in_file = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/train_raw_packed/grasps.csv')
    out_dir = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/train_raw_packed')

    # 读取CSV数据
    df = pd.read_csv(in_file)

    # # 分析原始数据中正样本和负样本的数量
    # positives = df[df["label"] == 1]
    # negatives = df[df["label"] == 0]
    # print("Number of samples:", len(df.index))
    # print("Number of positives:", len(positives.index))
    # print("Number of negatives:", len(negatives.index))
    # # 按照 scene_id, gripper_type, x, y, z 进行分组
    # pos_grouped = positives.groupby(['scene_id', 'gripper_type', 'x', 'y', 'z'])
    # neg_grouped = negatives.groupby(['scene_id', 'gripper_type', 'x', 'y', 'z'])
    # # 输出正负样本的数量
    # print("Number of merged positives:", len(pos_grouped))
    # print("Number of merged negatives:", len(neg_grouped))

    df_scn_grp = df.groupby(['scene_id'])

    for aug_type in tqdm(range(0, 4), desc="Overall Progress"):
        aug_data = []
        for name, group in tqdm(df_scn_grp, desc=f"Augmentation {aug_type}", leave=False):
            aug_data.append(augment_grasps(group, aug_type))
        aug_df = pd.concat(aug_data)
        aug_df.to_csv(out_dir / f'grasps_aug_{aug_type}.csv', index=False)