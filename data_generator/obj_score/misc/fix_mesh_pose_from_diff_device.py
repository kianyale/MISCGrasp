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


def update_npz_paths(file_path, new_prefix="/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data"):
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
    datas = np.load(file_path, allow_pickle=True)['pc']
    # print(datas)
    # raise
    for i, info in enumerate(datas):
        original_path = Path(info[0])  # 转为 Path 对象
        print(original_path)
        # 检查路径是否包含需要替换的前缀
        for prefix in ["/home/yons/MISCGrasp/data", "/ssd/MISCGrasp/data", "/ssddata/MISCGrasp/data", "data"]:  # , "/home/yons"]:
            if str(original_path).startswith(prefix):
                # 替换路径前缀为新前缀
                new_path = str(original_path).replace(prefix, new_prefix, 1)
                info[0] = new_path
                datas[i] = info
                print(info)
                print()
                break  # 找到匹配前缀后退出循环
    np.savez(file_path, pc=datas)
    return datas


if __name__ == "__main__":
    root = Path('data/datasets/data_vgn/washed/packed/mesh_pose_list')
    for file in root.glob('*.npz'):
        update_npz_paths(file)