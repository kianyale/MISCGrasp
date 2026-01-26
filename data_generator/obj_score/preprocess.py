# import os
# import time
#
# import yaml
# from collections import defaultdict
# import numpy as np
# from matplotlib import pyplot as plt
#
# # 定义子文件夹路径和GRIPPER_TYPES
# root_folder = "/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets"
# scales = ['small', 'medium', 'large']
# gripper_types = [
#     'franka', 'robotiq_2f_85', 'robotiq_2f_140', 'wsg_32', 'ezgripper', 'sawyer',
#     'wsg_50', 'rg2', 'barrett_hand_2f', 'kinova_3f', 'robotiq_3f', 'barrett_hand'
# ]
#
# # 使用 defaultdict 存储结果
# score_data = defaultdict(dict)
#
# # 遍历子文件夹
# for scale in scales:
#     folder_path = os.path.join(root_folder, f"{scale}_score_files", "score_raw")
#
#     # 遍历 GRIPPER_TYPES
#     for gripper in gripper_types:
#         yaml_file_path = os.path.join(folder_path, f"{gripper}_{scale}_score_raw.yaml")
#
#         # 检查 YAML 文件是否存在
#         if os.path.exists(yaml_file_path):
#             try:
#                 with open(yaml_file_path, 'r') as file:
#                     # 加载 YAML 文件内容
#                     data = yaml.full_load(file)
#                     if data:
#                         # 将数据转换为字典，避免元组不可变问题
#                         for key, value in data.items():
#                             data[key] = {
#                                 "complexity": value[0],
#                                 "difficulty": value[1]["difficulty"],
#                                 "num_negative": value[1]["num_negative"],
#                                 "num_positive": value[1]["num_positive"]
#                             }
#                         score_data[gripper].setdefault(scale, {}).update(data)
#             except yaml.YAMLError as e:
#                 print(f"Error parsing YAML file {yaml_file_path}: {e}")
#             except IOError as e:
#                 print(f"Error reading file {yaml_file_path}: {e}")
#
# # 对 complexity 和 difficulty 进行 Z-Score 标准化
# def z_score_normalize(gripper_data):
#     complexity_values = []
#     difficulty_values = []
#
#     # 收集所有值
#     for scale, objects in gripper_data.items():
#         for obj, metrics in objects.items():
#             complexity_values.append(metrics["complexity"])
#             difficulty_values.append(metrics["difficulty"])
#
#     # 转换为 numpy 数组
#     complexity_values = np.array(complexity_values)
#     difficulty_values = np.array(difficulty_values)
#
#     # 计算均值和标准差
#     complexity_mean, complexity_std = complexity_values.mean(), complexity_values.std()
#     difficulty_mean, difficulty_std = difficulty_values.mean(), difficulty_values.std()
#
#     for scale, objects in gripper_data.items():
#         for obj, metrics in objects.items():
#             metrics["zscore_complexity"] = float((metrics["complexity"] - complexity_mean) / complexity_std)
#             metrics["zscore_difficulty"] = float((metrics["difficulty"] - difficulty_mean) / difficulty_std)
#
# # 对每个夹爪的数据进行 Z-Score 标准化
# for gripper, gripper_data in score_data.items():
#     z_score_normalize(gripper_data)
#
# # 打印标准化后的结果
# for gripper, gripper_data in score_data.items():
#     print(f"\nGripper: {gripper}")
#     for scale, objects in gripper_data.items():
#         print(f"  Scale: {scale}")
#         for obj, metrics in objects.items():
#             print(f"    Object: {obj}, Z-Score Complexity: {metrics['zscore_complexity']:.2f}, Z-Score Difficulty: {metrics['zscore_difficulty']:.2f}")
#
# # 计算线性加权综合分数
# def compute_weighted_score(gripper_data, weight_complexity=0.3, weight_difficulty=0.7):
#     for scale, objects in gripper_data.items():
#         for obj, metrics in objects.items():
#             metrics["weighted_score"] = float(weight_complexity * metrics["zscore_complexity"] +
#                                           weight_difficulty * metrics["zscore_difficulty"])
#
# # 对每个夹爪的数据计算加权分数
# for gripper, gripper_data in score_data.items():
#     compute_weighted_score(gripper_data)
#
# # 打印加权分数的结果
# for gripper, gripper_data in score_data.items():
#     print(f"\nGripper: {gripper}")
#     for scale, objects in gripper_data.items():
#         print(f"  Scale: {scale}")
#         for obj, metrics in objects.items():
#             print(f"    Object: {obj}, Weighted Score: {metrics['weighted_score']:.2f}")
#
# # 重新划分等级并调整得分
# def rank_and_rescale(gripper_data):
#     # 收集所有物体及其得分，并添加尺寸信息
#     all_objects = []
#     for scale, objects in gripper_data.items():
#         for obj, metrics in objects.items():
#             unique_name = f"{scale}_{obj}"  # 添加尺寸作为唯一标识
#             all_objects.append((unique_name, metrics["weighted_score"], scale, obj))
#
#     # 按得分排序
#     all_objects.sort(key=lambda x: x[1])
#
#     # 划分为三个等级
#     low = all_objects[:39]
#     moderate = all_objects[39:108]
#     high = all_objects[108:]
#
#     # 对每个等级进行 Min-Max 归一化并调整得分为总和100
#     def normalize_and_rescale(group):
#         if not group:  # 检查组是否为空
#             return []
#         scores = np.array([score for _, score, _, _ in group])
#         min_score, max_score = scores.min(), scores.max()
#         normalized_scores = (scores + abs(min_score))
#         # normalized_scores = (scores - min_score) / (max_score - min_score + 1e-6)
#         normalized_scores = normalized_scores / normalized_scores.sum()
#         rescaled_scores = normalized_scores * 100
#         return [(unique_name, float(score), scale, obj) for (unique_name, _, scale, obj), score in zip(group, rescaled_scores)]
#
#     low = normalize_and_rescale(low)
#     moderate = normalize_and_rescale(moderate)
#     high = normalize_and_rescale(high)
#
#     # 更新原始数据中的得分，并存储等级标签
#     for unique_name, new_score, scale, obj in low:
#         if obj in gripper_data[scale]:
#             gripper_data[scale][obj]["final_score"] = new_score
#             gripper_data[scale][obj]["level"] = "low"
#
#     for unique_name, new_score, scale, obj in moderate:
#         if obj in gripper_data[scale]:
#             gripper_data[scale][obj]["final_score"] = new_score
#             gripper_data[scale][obj]["level"] = "moderate"
#
#     for unique_name, new_score, scale, obj in high:
#         if obj in gripper_data[scale]:
#             gripper_data[scale][obj]["final_score"] = new_score
#             gripper_data[scale][obj]["level"] = "high"
#
#     return gripper_data, low, moderate, high
#
#
# def visualize_scores(gripper, gripper_data, low, moderate, high):
#     # 收集分组后的数据
#     categories = {"low": [], "moderate": [], "high": []}
#     for scale, objects in gripper_data.items():
#         for obj, metrics in objects.items():
#             score = metrics["final_score"]
#             unique_name = f"{scale}_{obj}"  # 确保唯一标识
#             if unique_name in [o[0] for o in low]:
#                 categories["low"].append(score)
#             elif unique_name in [o[0] for o in moderate]:
#                 categories["moderate"].append(score)
#             elif unique_name in [o[0] for o in high]:
#                 categories["high"].append(score)
#
#     # 确保每个分组都有数据
#     if not any(categories.values()):
#         print(f"No data available for Gripper: {gripper}")
#         return
#
#     # 绘制箱线图以显示得分分布
#     plt.boxplot([categories["low"], categories["moderate"], categories["high"]],
#                 labels=["low", "moderate", "high"],
#                 patch_artist=True,
#                 boxprops=dict(facecolor="lightblue"),
#                 medianprops=dict(color="red"))
#     plt.xlabel("Categories")
#     plt.ylabel("Final Score")
#     plt.title(f"Score Distribution for Gripper: {gripper}")
#     plt.show()
#
#
# for gripper, gripper_data in score_data.items():
#     updated_gripper_data, low, moderate, high = rank_and_rescale(gripper_data)
#
#     visualize_scores(gripper, gripper_data, low, moderate, high)
#
#     # 保存为 YAML 文件
#     output_folder = os.path.join(root_folder, "score")
#     os.makedirs(output_folder, exist_ok=True)
#     output_file = os.path.join(output_folder, f"{gripper}.yaml")
#
#     with open(output_file, "w") as file:
#         yaml.dump(updated_gripper_data, file)
#
#     print(f"Scores for gripper '{gripper}' saved to '{output_file}'")
#
# # 打印调整后的结果
# for gripper, gripper_data in score_data.items():
#     print(f"\nGripper: {gripper}")
#     for scale, objects in gripper_data.items():
#         print(f"  Scale: {scale}")
#         for obj, metrics in objects.items():
#             print(f"    Object: {obj}, Rescaled Score: {metrics['final_score']:.2f}")
