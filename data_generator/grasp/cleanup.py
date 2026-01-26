import sys
import os
import warnings

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')
import argparse
from pathlib import Path
import numpy as np

from src.gd.io import *

INTERVAL = 40

def main(args):
    if args.flag == 'train':
        # 读取数据
        df = read_df(args.root)

        # 可选的过滤操作（取消注释根据需求启用）
        # filters = (df["x"] >= 0.) & (df["y"] >= 0.) & (df["z"] >= 0.) & \
        #           (df["x"] <= 0.4) & (df["y"] <= 0.4) & (df["z"] <= 0.4)
        # df = df[filters]

        # 获取文件列表并按创建时间排序
        scenes_path = args.root / 'sensor_data'
        file_list = sorted(
            os.listdir(scenes_path),
            key=lambda file: os.path.getctime(scenes_path / file)
        )#[-INTERVAL:]

        # 使用更高效的循环与过滤操作
        for f in file_list:
            f_name = f.split('.')[0]
            scene_data = df[df['scene_id'] == f_name]

            # 直接使用groupby避免重复索引
            for (gripper_type, scale), group in scene_data.groupby(['gripper_type', 'scale']):
                positives = group[group["label"] == 1]
                negatives = group[group["label"] == 0]

                # if len(positives) > 32:
                #     drop_indices = np.random.choice(positives.index, len(positives) - 32, replace=False)
                #     df = df.drop(drop_indices)
                # 如果负样本数量超出限制，随机下采样
                if len(negatives) > 144:
                    drop_indices = np.random.choice(negatives.index, len(negatives) - 144, replace=False)
                    df = df.drop(drop_indices)

                print(f'{f_name}_{gripper_type}_{scale} is done.')

        # 写入更新后的数据
        write_df(df, args.root)

        # print('`sensor_data` and `mesh_pose_list` begin to clean.')
        # sensor_data = df["scene_id"].values
        # for f in (args.root / "sensor_data").iterdir():
        #     if f.suffix == ".npz" and f.stem not in sensor_data:
        #         print("Removed", f)
        #         f.unlink()
        # for f in (args.root / "mesh_pose_list").iterdir():
        #     if f.suffix == ".npz" and f.stem not in sensor_data:
        #         print("Removed", f)
        #         f.unlink()
        # print('`sensor_data` and `mesh_pose_list` is done.')

    elif args.flag == 'test':
        df = read_df(args.root)
        df.drop(df[df["x"] < 0.01].index, inplace=True)
        df.drop(df[df["y"] < 0.01].index, inplace=True)
        df.drop(df[df["z"] < 0.01].index, inplace=True)
        df.drop(df[df["x"] > 0.39].index, inplace=True)
        df.drop(df[df["y"] > 0.39].index, inplace=True)
        df.drop(df[df["z"] > 0.39].index, inplace=True)

        idx = sorted(df['scene_id'].unique())[:5]
        for id in idx:
            gripper_type = df[df['scene_id'] == id]['gripper_type'].unique()
            for t in gripper_type:
                scale = df[df['scene_id'] == id][df['gripper_type'] == t]['scale'].unique()
                for s in scale:
                    df_child = df[df['scene_id'] == id][df['gripper_type'] == t][df['scale'] == s]
                    positives = df_child[df_child["label"] == 1]
                    negatives = df_child[df_child["label"] == 0]
                    n = len(negatives.index) - len(positives.index)
                    if n > 0:
                        i = np.random.choice(negatives.index, n, replace=False)
                        df = df.drop(i)
                    elif n < 0:
                        i = np.random.choice(positives.index, -n, replace=False)
                        df = df.drop(i)
                    print(f'{id}_{t}_{s} is done. ')
        write_df(df, args.root)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--flag", type=str, choices=['train', 'test'])
    args = parser.parse_args()
    main(args)
