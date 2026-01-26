from argparse import ArgumentParser
from pathlib import Path
import sys
import pandas as pd
import os
import numpy as np
from colorama import Fore

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')
from src.gd import io


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):  # TODO
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "sensor_data" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label


##############################
# Combine all trials
##############################
def main(args):
    gripper_type, expname = args.gripper_type, args.expname
    log_root_dir = args.log_root_dir
    root_path = (Path(args.log_root_dir) / 'exp_results' / expname).resolve()
    round_dir_list = sorted(os.listdir(root_path))

    combine_path = root_path.parent / f"{root_path.name}_combine"
    if not combine_path.exists():
        combine_path.mkdir()

    df = pd.DataFrame()
    for i in range(len(round_dir_list)):
        df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "grasps.csv"))
        df_round["round_id"] = i
        df = pd.concat([df, df_round])
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(combine_path, "grasps.csv"), index=False)

    df = pd.DataFrame()
    for i in range(len(round_dir_list)):
        df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "rounds.csv"))
        df_round["round_id"] = i
        df = pd.concat([df, df_round])
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(combine_path, "rounds.csv"), index=False)
    ##############################
    # Print Stat
    ##############################
    data = Data(combine_path)

    # First, we compute the following metrics for the experiment:
    # * **Success rate**: the ratio of successful grasp executions,
    # * **Percent cleared**: the percentage of objects removed during each round,
    try:
        print("Path:              ", Fore.GREEN, path := str(combine_path), Fore.RESET)
        print("Num grasps:        ", Fore.GREEN, num_grasps := data.num_grasps(), Fore.RESET)
        print("Success rate:      ", Fore.GREEN, SR := data.success_rate(), Fore.RESET)
        print("Percent cleared:   ", Fore.GREEN, DR := data.percent_cleared(), Fore.RESET)
        print("Planning time:     ", Fore.GREEN, IT := data.avg_planning_time(), Fore.RESET)
    except:
        print(f"{Fore.YELLOW}[W] Incomplete results, exit{Fore.RESET}")
        exit()
    ##############################
    # Calc first-time grasping SR
    ##############################
    sum_label = 0
    firstgrasp_fail_expidx_list = []
    for i in range(len(round_dir_list)):
        # print(i)
        df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "grasps.csv"))
        df = df_round.iloc[0:1, :]

        label = df[["label"]].to_numpy(np.float32)
        if label.shape[0] == 0:
            firstgrasp_fail_expidx_list.append(i)
            continue
        sum_label += label[0, 0]
        if label[0, 0] == 0:
            firstgrasp_fail_expidx_list.append(i)

    print("First grasp success rate: ", Fore.GREEN, FSR := sum_label / len(round_dir_list), Fore.RESET)
    print("First grasp fail:", Fore.GREEN, len(firstgrasp_fail_expidx_list), "/", len(round_dir_list), ", round id: ",
          firstgrasp_fail_expidx_list, Fore.RESET)

    if args.save:
        summary_file = os.path.join(combine_path, "summary_results.txt")
        with open(summary_file, 'w') as f:
            f.write("Experiment name:    " + str(expname) + '\n')
            f.write("Num grasps:         " + str(num_grasps) + '\n')
            f.write("Success rate:       " + str(SR) + '\n')
            f.write("Percent cleared:    " + str(DR) + '\n')
            f.write(f"Planning time:     " + str(IT) + '\n')
            f.write("First success rate: " + str(FSR) + '\n')
            f.write("First grasp fail:   " + str(len(firstgrasp_fail_expidx_list)) + '/' + str(len(round_dir_list)) + '\n')
            f.write("First grasp fail round id:   " + str(firstgrasp_fail_expidx_list) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("log_root_dir", type=str, help="Root directory for logs")
    parser.add_argument("expname", type=str, help="Experiment name")
    parser.add_argument("gripper_type", type=str, help="Experiment name")
    parser.add_argument("save", type=int, help="Save results to file")
    args = parser.parse_args()
    main(args)