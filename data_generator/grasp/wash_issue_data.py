import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append("/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/")

from src.gd.utils.transform import Transform, Rotation
from src.gd.io import *
from data_generator.grasp.gripper_module import load_gripper

parallel = ['franka', 'wsg_32', 'wsg_50', 'sawyer']
adaptive = ['robotiq_2f_85', 'robotiq_2f_140', 'ezgripper', 'rg2']
other = ['barrett_hand_2f']
tol = 2e-3

def is_valid(row) -> bool:
    label = row["label"]
    if label == 0:
        return True
    scene_id = row["scene_id"]
    gripper_type = row["gripper_type"]
    scale = row["scale"]
    orientation = Rotation.from_quat(row["qx":"qw"].to_numpy(np.double))
    position = row["x":"z"].to_numpy(np.double)
    if position[2] < 0.05:
        return False
    w_ratio = row["width"]

    func = load_gripper(gripper_type)
    gripper = func(None, scale)
    pose = Transform(orientation, position)
    w = gripper.max_opening_width * w_ratio

    if gripper_type in parallel or gripper_type in other:
        d = gripper.finger_depth
    elif gripper_type in adaptive:
        d = -(gripper.finger_depth - gripper.finger_depth_init) * w_ratio + gripper.finger_depth
    elif gripper_type in other:
        d = (gripper.finger_depth - gripper.finger_depth_close) * w_ratio + gripper.finger_depth_close
    else:
        raise NotImplementedError

    left_finger = pose * Transform(Rotation.identity(), [0., -w / 2, d])
    right_finger = pose * Transform(Rotation.identity(), [0., w / 2, d])
    if np.logical_and(left_finger.translation[2] > 0.05 - tol, right_finger.translation[2] > 0.05 - tol):
        return True
    else:
        return False


def process_csv(input_file, output_file):
    df = pd.read_csv(input_file / 'grasps.csv')
    valid_idx = df.apply(is_valid, axis=1)
    filtered_df = df[valid_idx]
    filtered_df.to_csv(output_file / 'grasps.csv', index=False)


if __name__ == '__main__':
    # input_path = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_multi_gripper/washed/train_raw_pile')
    input_path = Path('/home/yons/MISCGrasp/data_multi_gripper/washed/train_raw_pile_0630')
    output_path = Path('/home/yons/temp/train_raw_pile')

    if not output_path.exists():
        output_path.mkdir(parents=True)

    process_csv(input_path, output_path)
