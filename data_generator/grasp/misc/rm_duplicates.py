import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from colorama import Fore
from tqdm import tqdm
import sys
from multiprocessing import Pool, cpu_count
import scipy.signal as signal

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')
from src.gd.utils.transform import Transform, Rotation


def combine_grasps(group):
    """处理单个分组"""
    # 提取基础信息
    new_row = group.iloc[0].to_dict()
    # 提取四元数并计算方向
    orientations = Rotation.from_quat(group[['qx', 'qy', 'qz', 'qw']].to_numpy(np.double))
    z_axis = orientations.as_matrix()[:, :, 2]  # 提取旋转矩阵的 Z 轴
    norm_z_axis = z_axis / np.linalg.norm(z_axis, axis=-1, keepdims=True)

    # 检查 Z 轴方向的一致性
    if not np.allclose(norm_z_axis.mean(axis=0), norm_z_axis[0], atol=1e-4):
        print(f"{Fore.RED}Pay attention to {new_row['scene_id']}.")

    if new_row['label'] == 1:
        # 确定参考轴
        ref_axis = w_y_axis if np.isclose(np.abs(np.dot(w_x_axis, norm_z_axis[0])), 1.0, atol=1e-4) else w_x_axis

        # 计算新的基坐标系
        y_axis = np.cross(norm_z_axis[0], ref_axis)
        x_axis = np.cross(y_axis, norm_z_axis[0])
        R = Rotation.from_matrix(np.vstack((x_axis, y_axis, norm_z_axis[0])).T)

        # 生成 12 个 yaw 角度
        yaws = np.linspace(0.0, 2 * np.pi, 12)[:-1]
        ref_oris = R * Rotation.from_euler('z', yaws)

        # 初始化 inplane 标签
        results = np.zeros(11)
        widths = np.zeros(11)
        ori_widths = group['width'].to_numpy()
        j = 0
        for i, ref_ori in enumerate(ref_oris):
            ori_diffs = orientations.inv() * ref_ori
            angles = ori_diffs.magnitude()
            if np.any(np.abs(angles) < 1e-4):
                results[i] = 1
                widths[i] = ori_widths[j]
                j += 1

        if results.any():
            peaks, properties = signal.find_peaks(
                x=np.r_[0, results, 0],  # fill in 0 before and after the array
                height=1,
                width=1
            )
            idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1  # so here needs to minus 1

            ori = ref_oris[idx_of_widest_peak]
            width = widths[idx_of_widest_peak]
        else:
            ori = ref_oris[0]
            width = ori_widths[-1]

        new_row['qx'], new_row['qy'], new_row['qz'], new_row['qw'] = ori.as_quat()
        new_row['width'] = width

    return new_row


def process_group(args):
    """多进程处理单个分组"""
    name, group = args
    return combine_grasps(group)


def main():
    in_file = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/train_raw_pile') / 'grasps.csv'
    out_file = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/train_raw_pile') / 'grasps.csv'

    # 读取 CSV 数据
    print("Reading CSV...")
    df = pd.read_csv(in_file)
    # df = df[df['label'] == 1]

    # 按照 scene_id, gripper_type, x, y, z, label 分组
    df_scn_grp = list(df.groupby(['scene_id', 'gripper_type', 'x', 'y', 'z', 'label']))

    # 使用多进程处理分组
    print("Processing groups in parallel...")
    with Pool(processes=cpu_count()) as pool:
        inplane_data = list(tqdm(pool.imap(process_group, df_scn_grp), total=len(df_scn_grp)))

    # 合并结果并保存
    print("Saving results...")
    inplane_df = pd.DataFrame.from_records(inplane_data)
    inplane_df.to_csv(out_file, index=False, chunksize=100000)


if __name__ == "__main__":
    main()


