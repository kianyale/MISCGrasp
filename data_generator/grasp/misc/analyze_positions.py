# import math
# import pandas as pd
# from collections import defaultdict
# from colorama import Fore, Style
#
#
# def analyze_positions(csv_file):
#     # 读取 CSV 文件
#     df = pd.read_csv(csv_file)
#
#     # 将 i, j, k 四舍五入并限制在边界内
#     df['i'] = df['i'].round().clip(0, 79).astype(int)
#     df['j'] = df['j'].round().clip(0, 79).astype(int)
#     df['k'] = df['k'].round().clip(0, 79).astype(int)
#
#     # 使用字典存储统计信息，避免频繁的 DataFrame 操作
#     results = []
#     scene_gripper_groups = defaultdict(list)
#
#     # 按场景和夹爪类型分组
#     for _, row in df.iterrows():
#         scene_gripper_groups[(row['scene_id'], row['gripper_type'])].append(row)
#
#     # 处理每个场景和夹爪类型的组合
#     for (scene_id, gripper_type), group in scene_gripper_groups.items():
#         # 初始化计数器
#         positive_count = 0
#         negative_count = 0
#         same_position_count = 0  # 统计正负例同时存在相同位置的计数
#         pos_uniq_positions = defaultdict(int)
#         neg_uniq_positions = defaultdict(int)
#
#         # 使用字典存储每个位置的正负例数量
#         position_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
#
#         # 遍历当前组内的所有数据
#         for row in group:
#             i, j, k, label = row['i'], row['j'], row['k'], row['label']
#             position_key = (i, j, k)
#
#             # 更新每个位置的标签数量
#             if label == 1:
#                 position_counts[position_key]['positive'] += 1
#                 positive_count += 1
#             else:
#                 position_counts[position_key]['negative'] += 1
#                 negative_count += 1
#
#         # 计算同时有正负例的相同位置
#         for position_key, counts in position_counts.items():
#             pos_labels = counts['positive']
#             neg_labels = counts['negative']
#
#             # 如果同一位置既有正例又有负例，统计
#             if pos_labels > 0 and neg_labels > 0:
#                 same_position_count += 1
#
#             # 更新唯一位置计数
#             if pos_labels > 0:
#                 pos_uniq_positions[position_key] = 1
#             if neg_labels > 0:
#                 neg_uniq_positions[position_key] = 1
#
#         # 记录结果
#         # if same_position_count > 0:
#         results.append({
#             'scene_id': scene_id,
#             'gripper_type': gripper_type,
#             'same_position_count': same_position_count,  # 记录相同位置的正负例数量
#             'positive_total': positive_count,
#             'negative_total': negative_count,
#             'pos_uniq_positions': len(pos_uniq_positions),  # 记录唯一位置的数量
#             'neg_uniq_positions': len(neg_uniq_positions)  # 记录唯一位置的数量
#         })
#
#     # 打印结果
#     for result in results:
#         print(Fore.GREEN + f"Scene ID: {result['scene_id']}, Gripper Type: {result['gripper_type']}" + Style.RESET_ALL)
#         print(f"  Same Position Count (Positive and Negative): {result['same_position_count']}")
#         print(f"  Total Positive: {result['positive_total']}")
#         print(f"  Total Negative: {result['negative_total']}")
#         print(f"  Unique Positive Position Count: {result['pos_uniq_positions']}")
#         print(f"  Unique Negative Position Count: {result['neg_uniq_positions']}")
#         print("-" * 40)
#
#     print(f"  Sum: {len(results)}")
#     min_remain, max_same = math.inf, 0
#     count1 = 0
#     count2 = 0
#     for result in results:
#         if result['pos_uniq_positions'] < min_remain:
#             min_remain = result['pos_uniq_positions']
#
#         if result['same_position_count'] > max_same:
#             max_same = result['same_position_count']
#
#         if result['pos_uniq_positions'] > 8:
#             count1 += 1
#
#         if result['same_position_count'] > 0:
#             count2 += 1
#     print(f"  Min Remain: {min_remain}")
#     print(f"  pos_uniq_positions > 8: {count1}")
#     print(f"  Max Same: {max_same}")
#     print(f"  same_position_count > 0: {count2}")
#
#
# if __name__ == "__main__":
#     # 替换为你的 CSV 文件路径
#     csv_file = "/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_ga/processed/pile/grasps.csv"
#     analyze_positions(csv_file)
########################################################################################################################
import math
import pandas as pd
from collections import defaultdict
from colorama import Fore, Style
import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """将四元数转换为旋转矩阵"""
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return R[:, 2]  # 返回 z 轴向量

def analyze_positions(csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 将 i, j, k 四舍五入并限制在边界内
    df['i'] = df['i'].round().clip(0, 79).astype(int)
    df['j'] = df['j'].round().clip(0, 79).astype(int)
    df['k'] = df['k'].round().clip(0, 79).astype(int)

    # 使用字典存储统计信息，避免频繁的 DataFrame 操作
    results = []
    scene_gripper_groups = defaultdict(list)

    # 按场景和夹爪类型分组
    for _, row in df.iterrows():
        scene_gripper_groups[(row['scene_id'], row['gripper_type'])].append(row)

    # 处理每个场景和夹爪类型的组合
    for (scene_id, gripper_type), group in scene_gripper_groups.items():
        position_z_axes = defaultdict(list)
        inconsistent_z_axes_count = 0

        # 遍历当前组内的所有数据
        for row in group:
            i, j, k, label = row['i'], row['j'], row['k'], row['label']
            position_key = (i, j, k)

            if label == 1:  # 仅统计正例的情况
                z_axis = Rotation.from_quat(row['qx': 'qw']).as_matrix()[:, 2]
                position_z_axes[position_key].append(z_axis)

        # 检查同一位置的正例中，z 轴方向是否不一致
        for position_key, z_axes in position_z_axes.items():
            if len(z_axes) > 1:  # 该位置有多个正例
                # 归一化 z 轴向量
                z_axes = [z / np.linalg.norm(z) for z in z_axes]
                # 比较第一个 z 轴与其他 z 轴的方向
                reference_z = z_axes[0]
                for z in z_axes[1:]:
                    dot_product = np.dot(reference_z, z)
                    angle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
                    if abs(angle) > 2:  # 夹角大于 5 度认为方向不一致
                        inconsistent_z_axes_count += 1
                        break

        # 记录结果
        results.append({
            'scene_id': scene_id,
            'gripper_type': gripper_type,
            'inconsistent_z_axes_count': inconsistent_z_axes_count
        })

    # 打印结果
    for result in results:
        print(Fore.GREEN + f"Scene ID: {result['scene_id']}, Gripper Type: {result['gripper_type']}" + Style.RESET_ALL)
        print(f"  Inconsistent Z Axes Count: {result['inconsistent_z_axes_count']}")
        print("-" * 40)

if __name__ == "__main__":
    # 替换为你的 CSV 文件路径
    csv_file = "/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_ga/processed/pile/grasps.csv"
    analyze_positions(csv_file)


