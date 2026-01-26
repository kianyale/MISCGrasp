import json
import uuid

import numpy as np
import pandas as pd

from src.gd.grasp import Grasp
from src.gd.perception import *
from src.gd.utils.transform import Rotation, Transform


def write_setup(root, size, intrinsic):  # max_opening_width, finger_depth): TODO
    data = {
        "size": size,
        "intrinsic": intrinsic.to_dict()
    }
    write_json(data, root / "setup.json")


def read_setup(root):
    data = read_json(root / "setup.json")
    size = data["size"]
    intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
    return size, intrinsic


def write_sensor_data(root, depth_imgs, extrinsics):
    scene_id = uuid.uuid4().hex
    path = root / "sensor_data" / (scene_id + ".npz")
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id


def read_sensor_data(root, scene_id):
    data = np.load(root / "sensor_data" / (scene_id + ".npz"))
    return data["depth_imgs"], data["extrinsics"]


def write_grasp(root, scene_id, gripper_type, scale, grasp, label):  # TODO
    # NOTE concurrent writes could be an issue
    csv_path = root / "grasps.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "gripper_type", "scale", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, gripper_type, scale, qx, qy, qz, qw, x, y, z, width, label)


def read_grasp(df, i):
    scene_id = df.loc[i, "scene_id"]
    gripper_type = df.loc[i, "gripper_type"]
    scale = df.loc[i, "scale"]
    orientation = Rotation.from_quat(df.loc[i, "qx":"qw"].to_numpy(np.double))
    position = df.loc[i, "x":"z"].to_numpy(np.double)
    width = df.loc[i, "width"]
    label = df.loc[i, "label"]
    grasp = Grasp(Transform(orientation, position), width)
    return scene_id, gripper_type, scale, grasp, label


def read_df(root):
    return pd.read_csv(root / "grasps.csv")


def write_df(df, root):
    df.to_csv(root / "grasps.csv", index=False)


def write_voxel_grid(root, scene_id, voxel_grid):
    path = root / (scene_id + ".npz")
    np.savez_compressed(path, grid=voxel_grid)


def write_voxel_grid_aug(root, scene_id, num_id, voxel_grid):
    path = root / scene_id / f"{num_id:04d}.npz"
    np.savez_compressed(path, grid=voxel_grid)


def read_voxel_grid(root, scene_id):
    path = root / "scenes" / (scene_id + ".npz")
    return np.load(path)["grid"]


def read_voxel_grid_aug(root, scene_id, num_id):
    path = root / "scenes" / scene_id / f"{num_id:04d}.npz"
    return np.load(path)["grid"]

def read_occ(root, scene_id, num_point):
    occ_paths = list((root / 'occ' / scene_id).glob('*.npz'))
    path_idx = np.random.randint(low=0, high=len(occ_paths), dtype=int)
    occ_path = occ_paths[path_idx]
    occ_data = np.load(occ_path)
    points = occ_data['points']
    occ = occ_data['occ']
    points, idxs = sample_point_cloud(points, num_point, return_idx=True)
    occ = occ[idxs]
    return points, occ

def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]

def read_gripper_grid(root, gripper_type, scale):
    path = root / "grippers_tsdf" / (f'{gripper_type}_{scale:.1f}' + ".npz")
    tsdf = np.load(path)
    return np.concatenate([tsdf['open'], tsdf['close']], axis=0)


def read_json(path):
    with path.open("r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def create_csv(path, columns):
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")
