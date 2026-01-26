import sys

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.gd.io import *
from src.gd.perception import *

RESOLUTION = 80
SIZE = 0.4


def main(args):
    # create directory of new dataset
    (args.dataset / "scenes").mkdir(parents=True)

    # load setup information
    size, intrinsic = read_setup(args.raw)
    assert np.isclose(size, SIZE)
    voxel_size = size / RESOLUTION

    # create df
    # df = read_df(args.raw)
    df = pd.read_csv(args.raw / 'grasps.csv')
    df["x"] /= voxel_size
    df["y"] /= voxel_size
    df["z"] /= voxel_size
    # df["width"] /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k"})
    # write_df(df, args.dataset)
    df.to_csv(args.dataset / "grasps.csv", index=False)

    # create tsdfs
    for f in tqdm(list((args.raw / "sensor_data").iterdir())):
        if f.suffix != ".npz":
            continue
        depth_imgs, extrinsics = read_sensor_data(args.raw, f.stem)

        # random aug
        # n = np.random.choice(6) + 1
        # ids = np.random.choice(6, size=n, replace=False)
        # depth_imgs, extrinsics = depth_imgs[ids], extrinsics[ids]

        tsdf = create_tsdf(size, RESOLUTION, depth_imgs, intrinsic, extrinsics)
        grid = tsdf.get_grid()
        write_voxel_grid(args.dataset / 'sensor_data', f.stem, grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    main(args)
