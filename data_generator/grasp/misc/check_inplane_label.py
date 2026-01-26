import math
from pathlib import Path

import numpy as np
import pandas as pd
from colorama import *
from tqdm import tqdm
import sys
sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')
from src.gd.utils.transform import Transform, Rotation

gripper_types = [
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


if __name__ == "__main__":
    # 文件路径
    in_file_inplane = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/data_trial/train_raw_packed/inplane_aug_0.csv')
    in_file_ori = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/data_trial/train_raw_packed/aug_0.csv')

    # 读取 CSV 数据
    df_inplane = pd.read_csv(in_file_inplane)
    df_ori = pd.read_csv(in_file_ori)

    # 仅保留 label 为 1 的数据
    df_ori = df_ori[df_ori['label'] == 1]

    # 按照 scene_id, gripper_type, x, y, z 分组
    check = {}
    # df_ori_scn_grp = df_ori.groupby(['scene_id', 'gripper_type', 'x', 'y', 'z'])

    # print(len(df_inplane[df_inplane['inplane_0'] != df_inplane['inplane_10']]))
    # raise

    # 检查 df_inplane 中的分组
    df_inplane_scn_grp = df_inplane.groupby(['scene_id']) #, 'x', 'y', 'z'])
    cnt = 0
    cnt_miss = 0
    for name, group in tqdm(df_inplane_scn_grp):
        flag = False
        for grp in gripper_types:
            try:
                if len(group[group['gripper_type'] == grp]) < 8:
                    flag = True
            except KeyError:
                cnt_miss += 1
                flag = True
                break
        if not flag:
            cnt += 1
    print(cnt, cnt_miss)
    raise

    # minn = math.inf
    # count_8 = 0
    # count_12 = 0
    # count_16 = 0
    # count_othr = 0
    # # 计算每个分组的标签数量
    # for name, group in tqdm(df_inplane_scn_grp, desc="Processing ori data"):
    #     # check[name] = len(group)  # 直接计算每个分组的大小
    #     minn = min(minn, len(group))
    #     if len(group) < 8:
    #         count_8 += 1
    #     elif len(group) < 12:
    #         count_12 += 1
    #     elif len(group) < 16:
    #         count_16 += 1
    #     else:
    #         count_othr += 1
    #
    # print(minn, count_8, count_12, count_16, count_othr)
    # raise

    count = 0
    # 逐组比较 df_inplane 和 df_ori 的标签数量
    for name, group in tqdm(df_inplane_scn_grp, desc="Processing inplane data"):
        # 获取 inplane_0 到 inplane_11 列的总和
        inplane_columns = [f'inplane_{i}' for i in range(11)]  # inplane_0 到 inplane_11 列
        count_inplane = group[inplane_columns].sum().sum()  # 计算这些列的总和
        if name in check:
            count_ori = check[name]  # 获取对应分组的标签数量
            if count_inplane != count_ori:
                count += 1
                print(f"Checking {Fore.RED} {name} {Fore.RESET}: inplane count ({count_inplane}) != ori count ({count_ori}).")
        else:
            print(f"Missing {Fore.RED} {name} {Fore.RESET} in ori data.")

    print(f'Sum: {count}.')



