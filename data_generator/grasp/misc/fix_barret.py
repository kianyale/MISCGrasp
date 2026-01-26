import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')
from src.gd.utils.transform import *

# Input and output paths
input_path = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/Backup/train_raw_packed/grasps.csv')
output_path = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/Backup/train_raw_packed/grasps.csv')

# Read the CSV file
df = pd.read_csv(input_path)

# Transformation matrix: move {'barrett_hand_2f': 0.027, 'barrett_hand': 0.008} meters along the negative Z-axis of the gripper coordinate system
delta_translation = np.array([0, 0, -0.008])


# Define the transformation function
def transform_pose(row):
    # Extract the original TCP position and quaternion orientation
    pose = Transform(Rotation.from_quat(row[['qx', 'qy', 'qz', 'qw']].values), row[['x', 'y', 'z']].values)
    transform = Transform(Rotation.identity(), delta_translation)

    # Calculate the new pose
    pose_new = pose * transform

    # Update the row data
    row[['qx', 'qy', 'qz', 'qw']] = pose_new.rotation.as_quat()
    row[['x', 'y', 'z']] = pose_new.translation
    return row


# Filter and modify the data
mask = (df['gripper_type'] == 'barrett_hand')
df.loc[mask] = df.loc[mask].apply(transform_pose, axis=1)

# Save the modified file
df.to_csv(output_path, index=False)

# Print part of the modified data for confirmation
# print(df[df['gripper_type'] == 'barrett_hand'].head())
