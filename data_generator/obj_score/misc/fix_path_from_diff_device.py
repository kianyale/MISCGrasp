from pathlib import Path
import pandas as pd
import numpy as np

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
gripper_scales = ['small', 'medium', 'large']

def list_all_subfolders(directory):
    directory = Path(directory)
    subfolders = []
    for item in directory.iterdir():
        if item.is_dir():
            subfolders.append(item)  # 添加当前子文件夹
            subfolders.extend(list_all_subfolders(item))  # 递归遍历
    return subfolders


def update_npy_paths(file_path, new_prefix="/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data"):
    """
    读取 .npy 文件内容，将路径前缀统一替换为指定的新前缀，并保存回文件。

    Args:
        file_path (str or Path): .npy 文件路径。
        new_prefix (str): 替换的新路径前缀，例如 '/ssd'。

    Returns:
        updated_data: 修改后的数据。
    """
    file_path = Path(file_path)

    # 读取 .npy 文件内容
    data = np.load(file_path, allow_pickle=True).item()

    # 遍历字典并更新路径前缀
    for key, value in data.items():
        if isinstance(value, list) and isinstance(value[-1], str):  # 确保最后一项是路径字符串
            original_path = Path(value[-1])  # 转为 Path 对象
            print(original_path)
            # 检查路径是否包含需要替换的前缀
            for prefix in ["data"]: #, "/home/yons"]:
                if str(original_path).startswith(prefix):
                    # 替换路径前缀为新前缀
                    new_path = str(original_path).replace(prefix, new_prefix, 1)
                    value[-1] = new_path
                    print(value)
                    break  # 找到匹配前缀后退出循环

    # 保存回文件
    np.save(file_path, data)
    return data


if __name__ == "__main__":
    root = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets/data/mesh_pose_list')
    paths = []
    for gripper_type in gripper_types:
        for gripper_scale in gripper_scales:
            paths.append(f'{gripper_type}_single_egad_{gripper_scale}_49')
    for path in paths:
        dir = root / path
        for file in dir.glob('*.npy'):
            update_npy_paths(file)