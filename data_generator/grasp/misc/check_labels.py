import csv
import pandas as pd
from collections import defaultdict
from colorama import Fore, Style

# 设置阈值
POSITIVE_THRESHOLD = 16
NEGATIVE_THRESHOLD = 0

# 文件名（请替换为实际文件路径）
CSV_FILE = "/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_ga/processed/pile/grasps.csv"
OUT_CSV_FILE = "/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_ga/processed/pile/grasps.csv"

gripper_types = [
  'franka',
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
  # 'barrett_hand'
]

def check_labels(csv_file):
    # 数据结构：{scene_id: {gripper_type: {'positive': count, 'negative': count}}}
    label_stats = defaultdict(lambda: defaultdict(lambda: {'positive': 0, 'negative': 0}))

    # 读取CSV文件
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            scene_id = row['scene_id']
            gripper_type = row['gripper_type']
            label = int(float(row['label']))

            if label == 1:
                label_stats[scene_id][gripper_type]['positive'] += 1
            elif label == 0:
                label_stats[scene_id][gripper_type]['negative'] += 1

    # 检查统计结果
    for scene_id, grippers in label_stats.items():
        for gripper_type, counts in grippers.items():
            positive_count = counts['positive']
            negative_count = counts['negative']

            # 检查是否低于阈值并输出结果
            if positive_count < POSITIVE_THRESHOLD or negative_count < NEGATIVE_THRESHOLD:
                print(Fore.RED + f"WARNING: Scene {scene_id}, Gripper {gripper_type} "
                      f"has low label counts: {positive_count} positive, {negative_count} negative." + Style.RESET_ALL)
            # else:
            #     print(f"Scene {scene_id}, Gripper {gripper_type}: "
            #           f"{positive_count} positive, {negative_count} negative.")

def wash_bad(csv_file):
    df = pd.read_csv(csv_file)
    cleaned_df = df[df['scene_id'].notnull() & (df['scene_id'] != "")]

    num_removed = len(df) - len(cleaned_df)
    if num_removed > 0:
        print(f"Removed {num_removed} rows with invalid scene_id.")
    else:
        print("No invalid scene_id found.")
    df.to_csv(OUT_CSV_FILE, index=False)

if __name__ == "__main__":
    pass
    # check_labels(CSV_FILE)
    # wash_bad(CSV_FILE)
    # df = pd.read_csv(CSV_FILE)
    # df = df[df['scene_id'] == '768103dfef2541b1aff4b11ca3e1bdfd']
    # for grp in gripper_types:
    #     print(df[df['gripper_type'] == grp])


# import pandas as pd
# import numpy as np
# from math import degrees, acos
# from scipy.spatial.transform import Rotation
#
# # 加载CSV文件
# file_path = '/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_ga/processed/packed/grasps.csv'  # 请替换为你的文件路径
# df = pd.read_csv(file_path)
#
#
# # 定义函数将四元数转换为旋转矩阵
# def quaternion_to_rotation_matrix(qx, qy, qz, qw):
#     R = np.array([
#         [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
#         [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
#         [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
#     ])
#     return R
#
#
# # 计算角度和标记
# def calculate_angle_and_label(row):
#     # 获取四元数
#     # qx, qy, qz, qw = row['qx'], row['qy'], row['qz'], row['qw']
#     R = Rotation.from_quat(row['qx': 'qw'])
#
#     # 计算旋转矩阵
#     # R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
#
#     # 提取抓取坐标系的 z 轴方向（第三列）
#     z_axis = R.as_matrix()[:, 2]  # [R[0,2], R[1,2], R[2,2]]
#
#     # 世界坐标系的 x 轴为 [1, 0, 0]
#     world_x_axis = np.array([1, 0, 0])
#
#     # 计算夹角（余弦定理）
#     dot_product = np.dot(z_axis, world_x_axis)
#     angle_rad = acos(np.clip(dot_product, -1.0, 1.0))  # 确保在合法范围内
#     angle_deg = degrees(angle_rad)
#
#     # 判断是否近似为 0 度或 180 度
#     is_near_0_or_180 = 1 if (abs(angle_deg) <= 1 or abs(angle_deg - 180) <= 1) else 0
#
#     return angle_deg, is_near_0_or_180
#
#
# # 应用计算函数
# df[['angle_z_to_x', 'near_0_or_180']] = df.apply(calculate_angle_and_label, axis=1, result_type='expand')
#
# # 统计标签数量
# label_count = df['near_0_or_180'].sum()
#
# # 输出结果
# print("计算完成！统计结果如下：")
# print(f"角度近似等于 0 度或 180 度的标签数量: {label_count}")
# print(df[['scene_id', 'gripper_type', 'angle_z_to_x', 'near_0_or_180']])

# 可选：保存到新CSV文件
# df.to_csv('output_with_angles.csv', index=False)
