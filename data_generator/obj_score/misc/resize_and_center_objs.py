import trimesh
import os
from pathlib import Path

# 文件夹路径
folder_path = '/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/egad/egadevalset/egad_eval_set'  # 替换为你的文件夹路径
save_path = '/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/object2urdf/ws/egad_eval_set_scale'  # 替换为你的文件夹路径


# 获取所有OBJ文件
obj_files = [f for f in os.listdir(folder_path) if f.endswith('.obj')]

for obj_file in obj_files:
    # 加载OBJ模型
    file_path = os.path.join(folder_path, obj_file)
    mesh = trimesh.load(file_path)

    # 尺寸缩小1000倍
    mesh.apply_scale(0.001)  # NOTE： put this before translation

    # 获取几何中心
    center = mesh.center_mass
    # print(center)

    # 移动中心到几何中心
    translation = -center
    # print(translation)
    mesh.apply_translation(translation)

    # 保存修改后的模型
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    new_file_path = os.path.join(save_path, obj_file)
    mesh.export(new_file_path)

    print(f"处理完成: {new_file_path}")
