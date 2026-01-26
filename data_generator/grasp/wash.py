import shutil
import sys
import os
import warnings
from colorama import Fore, Style

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')
import argparse
from pathlib import Path
import numpy as np

from src.gd.io import *


def main(args):
    if args.flag == 'train':
        df = read_df(args.root)

        # Filter data based on x, y, z ranges
        conditions = (df["x"] > 0) & (df["x"] < 0.4) & (df["y"] > 0) & (df["y"] < 0.4) & (df["z"] > 0) & (
                    df["z"] < 0.4)
        # # conditions = (df["i"] > 0) & (df["i"] < 80) & (df["j"] > 0) & (df["j"] < 80) & (df["k"] > 0) & (
        # #             df["k"] < 80)
        # df = df[conditions]

        # Remove rows with null or empty scene_id
        df = df[df["scene_id"].notnull() & (df["scene_id"] != "")]

        scene_ids = df["scene_id"].unique()
        if len(scene_ids) > args.num_scenes:
            sampled_scene_ids = pd.Series(scene_ids).sample(n=args.num_scenes, random_state=42).tolist()
            df = df[df['scene_id'].isin(sampled_scene_ids)]

        # Define the target gripper types
        gripper_types = [
            'franka',
            # 'robotiq_2f_85',
            # 'robotiq_2f_140',
            # 'wsg_32',
            # 'ezgripper',
            # 'sawyer',
            # 'wsg_50',
            # 'rg2',
            # 'barrett_hand_2f',
            # 'kinova_3f',
            # 'robotiq_3f',
            # 'barrett_hand'
        ]

        # Process the list of files
        file_list = os.listdir(args.root / 'sensor_data')
        trunc_file_list = sorted(file_list, key=lambda file: os.path.getctime(args.root / 'sensor_data' / file))

        print(f"{Fore.YELLOW}Processing sensor_data for cleaning...{Style.RESET_ALL}")

        for f in trunc_file_list:
            f_name = Path(f).stem
            df_scene = df[df['scene_id'] == f_name]

            # Check if all required gripper types are present in the scene
            missing_grippers = [g for g in gripper_types if g not in df_scene['gripper_type'].unique()]
            if missing_grippers:
                print(f"{Fore.RED}Scene {f_name} has been removed due to missing gripper types: {missing_grippers}{Style.RESET_ALL}")
                df = df[df['scene_id'] != f_name]  # Remove data for the scene
                continue

            is_dropped = False

            for t in df_scene['gripper_type'].unique():
                df_gripper = df_scene[df_scene['gripper_type'] == t]
                for s in df_gripper['scale'].unique():
                    df_child = df_gripper[df_gripper['scale'] == s]
                    positives = df_child[df_child["label"] == 1]
                    negatives = df_child[df_child["label"] == 0]
                    if len(positives) < 8 or len(negatives) < 16:
                        df = df[df['scene_id'] != f_name]
                        is_dropped = True
                        break
                    if len(negatives) > 160:
                        drop_indices = np.random.choice(negatives.index, len(negatives) - 160, replace=False)
                        df = df.drop(drop_indices)
                        print(f"Scene {f_name} removed {Fore.GREEN} {len(negatives) - 160} negatives.{Style.RESET_ALL}")

            if is_dropped:
                print(f"{Fore.RED}Scene {f_name} has been removed due to insufficient data.{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}Scene {f_name} passed all checks.{Style.RESET_ALL}")

        # Clean files in `sensor_data` and `mesh_pose_list`
        print(f"{Fore.YELLOW}Cleaning files in `sensor_data` and `mesh_pose_list`...{Style.RESET_ALL}")
        scenes = set(df["scene_id"].unique())

        for f in (args.root / "scenes").iterdir():
            if f.suffix == ".npz" and f.stem not in scenes:
                print(f"{Fore.RED}Removed {f.name} from `scenes`.{Style.RESET_ALL}")
                f.unlink()

        for f in (args.root / "sensor_data").iterdir():
            if f.suffix == ".npz" and f.stem not in scenes:
                print(f"{Fore.RED}Removed {f.name} from `sensor_data`.{Style.RESET_ALL}")
                f.unlink()

        for f in (args.root / "mesh_pose_list").iterdir():
            if f.suffix == ".npz" and f.stem not in scenes:
                print(f"{Fore.RED}Removed {f.name} from `mesh_pose_list`.{Style.RESET_ALL}")
                f.unlink()

        # For `GIGA` data
        if (occ_path:= args.root / "occ").exists():
            for f in occ_path.iterdir():
                if f.is_dir() and f.name not in scenes:
                    print(f"{Fore.RED}Removed folder {f.name} from `mesh_pose_list`.{Style.RESET_ALL}")
                    shutil.rmtree(f)

        # Write the cleaned data back to disk
        write_df(df, args.root)

        print(f"{Fore.GREEN}`sensor_data` and `mesh_pose_list` cleaning completed!{Style.RESET_ALL}")

    elif args.flag == 'test':
        df = read_df(args.root)
        df.drop(df[df["x"] < 0.01].index, inplace=True)
        df.drop(df[df["y"] < 0.01].index, inplace=True)
        df.drop(df[df["z"] < 0.01].index, inplace=True)
        df.drop(df[df["x"] > 0.39].index, inplace=True)
        df.drop(df[df["y"] > 0.39].index, inplace=True)
        df.drop(df[df["z"] > 0.39].index, inplace=True)

        idx = sorted(df['scene_id'].unique())[:5]
        for id in idx:
            gripper_type = df[df['scene_id'] == id]['gripper_type'].unique()
            for t in gripper_type:
                scale = df[df['scene_id'] == id][df['gripper_type'] == t]['scale'].unique()
                for s in scale:
                    df_child = df[df['scene_id'] == id][df['gripper_type'] == t][df['scale'] == s]
                    positives = df_child[df_child["label"] == 1]
                    negatives = df_child[df_child["label"] == 0]
                    n = len(negatives.index) - len(positives.index)
                    if n > 0:
                        i = np.random.choice(negatives.index, n, replace=False)
                        df = df.drop(i)
                    elif n < 0:
                        i = np.random.choice(positives.index, -n, replace=False)
                        df = df.drop(i)
                    print(f'{id}_{t}_{s} is done. ')
        write_df(df, args.root)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--flag", type=str, choices=['train', 'test'])
    parser.add_argument("--num_scenes", type=int, default=1000000000)
    args = parser.parse_args()
    main(args)
