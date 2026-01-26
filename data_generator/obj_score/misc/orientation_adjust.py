import trimesh
from pathlib import Path
import numpy as np
from data_generator.obj_score.obj_evaluate import ObjEvaluator

"""
A0, B0, B1, B2, B3, B6, C1, D0, E0, E6, F4, G4
B5, E2, E3, D4, E4, F1, F3, F4
G2, G6, F6, G5, F0, F3, G3
"""

# p1 = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/object2urdf/ws/E4.obj')
# H1 = ObjEvaluator.compute_complexity(p1)
# p_1 = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/object2urdf/ws/egad_eval_set_scale/E4.obj')
# H_1 = ObjEvaluator.compute_complexity(p_1)
# print(H1, H_1)
# raise

in_obj_path = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/object2urdf/ws/egad_eval_set_scale/G3.obj')
out_obj_path = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/object2urdf/ws/G3.obj')

# 加载模型
mesh = trimesh.load_mesh(in_obj_path)  # 替换为你模型的路径

# 显示原始物体
mesh.show()

# # 1. 平移物体
# translation = np.array([1.0, 0.0, 0.0])  # 在X轴方向平移1个单位
# mesh.apply_translation(translation)

# 沿 X 轴缩放，其他方向保持不变
# scale_factor = np.array([1, 1.5, 1])
# mesh.apply_scale(scale_factor)

# 2. 旋转物体
# 旋转矩阵：绕Z轴旋转45度
rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
mesh.apply_transform(rotation_matrix)
# rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
# mesh.apply_transform(rotation_matrix)

# 3. 再次显示调整后的物体
mesh.show()

mesh.export(out_obj_path)  # 保存为 .obj 文件，替换为保存的路径