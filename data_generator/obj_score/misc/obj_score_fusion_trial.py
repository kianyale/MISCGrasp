import numpy as np
import pandas as pd

# 假设抓取困难度和几何复杂度分别如下
# np.random.seed(42)  # 固定随机种子
# geometries = np.random.uniform(0.1, 0.6, 49 * 3)  # 几何复杂度 (0.1 ~ 0.6)
# grasp_difficulties = np.random.uniform(0.75, 1.0, 49 * 3)  # 抓取困难度 (0.75 ~ 1.0)
#
# # 归一化处理
# geometries_normalized = (geometries - geometries.min()) / (geometries.max() - geometries.min())
# grasp_difficulties_normalized = (grasp_difficulties - grasp_difficulties.min()) / (grasp_difficulties.max() - grasp_difficulties.min())
#
# # 融合分数计算
# fusion_scores = 0.7 * grasp_difficulties_normalized + 0.3 * geometries_normalized
#
# # 分数归一化到 100 分范围
# final_scores = fusion_scores / fusion_scores.sum() * 100
#
# # 转为 DataFrame 便于观察
# df = pd.DataFrame({
#     'Geometry': geometries,
#     'GraspDifficulty': grasp_difficulties,
#     'FusionScore': fusion_scores,
#     'FinalScore': final_scores
# })
#
# # 按 FinalScore 排序
# df = df.sort_values(by='FinalScore', ascending=False)
#
# # 打印前10个结果
# print(df)
#
# # 总分验证
# print("Total Score:", final_scores.sum())  # 应该接近 100

###########################################################

# import numpy as np
# import pandas as pd
#
# # 假设抓取困难度和几何复杂度
# np.random.seed(42)  # 固定随机种子
# geometries = np.random.uniform(0.1, 0.6, 49 * 3)  # 几何复杂度 (0.1 ~ 0.6)
# grasp_difficulties = np.random.uniform(0.75, 1.0, 49 * 3)  # 抓取困难度 (0.75 ~ 1.0)
#
# # 中心化处理（正负值分布）
# geometries_centered = 2 * ((geometries - geometries.min()) / (geometries.max() - geometries.min()) - 0.5)
# grasp_difficulties_centered = 2 * ((grasp_difficulties - grasp_difficulties.min()) / (grasp_difficulties.max() - grasp_difficulties.min()) - 0.5)
#
# # 融合分数计算（加权中心化值）
# fusion_scores = 0.7 * grasp_difficulties_centered + 0.3 * geometries_centered
#
# # 归一化到总分为 100
# final_scores = fusion_scores / np.sum(np.abs(fusion_scores)) * 100
#
# # 转为 DataFrame 便于观察
# df = pd.DataFrame({
#     'Geometry': geometries,
#     'GraspDifficulty': grasp_difficulties,
#     'GeometryCentered': geometries_centered,
#     'GraspDifficultyCentered': grasp_difficulties_centered,
#     'FusionScore': fusion_scores,
#     'FinalScore': final_scores
# })
#
# # 按 FinalScore 排序
# df = df.sort_values(by='FinalScore', ascending=False)
#
# # 打印前10个结果
# print(df.head(10))
#
# # 总分验证
# print("Total Score (absolute sum):", np.sum(np.abs(final_scores)))  # 应该接近 100
# print("Total Score (sum):", np.sum(final_scores))  # 应该接近 0

###########################################################

import numpy as np
import pandas as pd

# 假设每个物体已经有一个分数
np.random.seed(42)
scores = np.random.uniform(-1, 1, 147)  # 147 个物体的分数（预处理数据，范围 [-1, 1]）

# 定义物体类别
num_easy = 29
num_normal = 89
num_hard = 29

# 按分数将物体分为三类
sorted_indices = np.argsort(scores)
easy_indices = sorted_indices[:num_easy]
normal_indices = sorted_indices[num_easy:num_easy + num_normal]
hard_indices = sorted_indices[num_easy + num_normal:]

# 初始化得分容器
final_scores = np.zeros_like(scores)

# 分数分配逻辑
# 确保简单物体未抓起扣分
easy_negative = -100 / num_easy
# 普通物体抓起来得分
normal_positive = 100 / num_normal
# 困难物体抓起来得分
hard_positive = 100 / num_hard

# 为每类物体分配分数
final_scores[easy_indices] = easy_negative  # 未抓起扣分，抓起来得 0 分
final_scores[normal_indices] = normal_positive  # 抓起来得分，未抓起不得分
final_scores[hard_indices] = hard_positive  # 抓起来得分，未抓起不扣分

# 转换为 DataFrame 便于操作
df = pd.DataFrame({
    'ObjectID': np.arange(1, 148),
    'InitialScore': scores,
    'Category': ['Easy'] * num_easy + ['Normal'] * num_normal + ['Hard'] * num_hard,
    'AssignedScore': final_scores
})

# 打印结果验证分配逻辑
print(df.head(149))


# 仿真过程分数更新函数
def update_score(object_id, success, df):
    """
    根据抓取结果更新分数。
    :param object_id: 物体 ID
    :param success: 是否抓取成功 (True/False)
    :param df: 数据表
    """
    category = df.loc[df['ObjectID'] == object_id, 'Category'].values[0]
    assigned_score = df.loc[df['ObjectID'] == object_id, 'AssignedScore'].values[0]

    # 根据类别和抓取结果更新分数
    if category == 'Easy':
        return 0 if success else assigned_score
    elif category == 'Normal':
        return assigned_score if success else 0
    elif category == 'Hard':
        return assigned_score if success else 0


# 仿真测试
total_score = 0
simulation_results = [True] * 50 + [False] * 97  # 模拟抓取结果
for i, result in enumerate(simulation_results):
    total_score += update_score(i + 1, result, df)

print("Final Total Score:", total_score)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(len(final_scores)), final_scores)
plt.xlabel("Objects")
plt.ylabel("Final Score")
plt.title("Distribution of Final Scores")
plt.show()

# 检查所有物体抓起来时的总分
total_score_if_all_grasped = df['AssignedScore'].where(df['Category'] != 'Easy', 0).sum()
print("Total Score if all objects are grasped:", total_score_if_all_grasped)

