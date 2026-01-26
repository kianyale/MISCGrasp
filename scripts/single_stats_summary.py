from pathlib import Path
import sys

from colorama import *
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
import warnings

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.rounds = pd.read_csv(log_dir / "rounds.csv")
        self.pre_grasps = pd.read_csv(log_dir / "pre_grasps.csv")
        self.exc_grasps = pd.read_csv(log_dir / "exc_grasps.csv")
        self.levels = ['low', 'moderate', 'high']  # Attribute variable for levels

    def num_rounds(self):
        """Return the total number of rounds."""
        return len(self.rounds.index)

    def num_grasps(self):
        """Return the total number of grasp attempts."""
        return len(self.pre_grasps.index)

    def success_rate(self):
        """
        Calculate the success rate for each level and overall.
        Success rate is defined as the mean of the 'label' column.
        """
        success_rates = {level: self.exc_grasps[self.exc_grasps['level'] == level]["label"].mean() * 100
                         for level in self.levels}
        success_rates['total'] = self.exc_grasps["label"].mean() * 100
        return success_rates

    def percent_cleared(self):
        """
        Calculate the percentage of objects cleared for each level and overall.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleared_rates = {}
            for level in self.levels:
                filtered_exc_grasps = self.exc_grasps[self.exc_grasps['level'] == level]
                cleared_counts = filtered_exc_grasps.groupby("round_id")["label"].sum()
                merged = self.rounds[self.rounds['level'] == level].set_index("round_id").join(cleared_counts.rename("cleared_count"), how="left").fillna(0)
                cleared_rates[level] = (merged["cleared_count"].sum() / merged["object_count"].sum()) * 100

            cleared_counts = self.exc_grasps.groupby("round_id")["label"].sum()
            merged = self.rounds.set_index("round_id").join(cleared_counts.rename("cleared_count"), how="left").fillna(0)
            cleared_rates['total'] = (merged["cleared_count"].sum() / merged["object_count"].sum()) * 100
            return cleared_rates

    def avg_planning_time(self):
        """Calculate the average inference time."""
        return self.exc_grasps["planning_time"].mean()

    def eval_score(self):
        """
        Calculate the evaluation score.
        For each round_id, the score is the sum of 'eval_score' for successful grasps divided by
        the total number of attempts for that round_id.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_scores = {}

            # Calculate for each level
            for level in self.levels:
                filtered = self.exc_grasps[self.exc_grasps['level'] == level]
                if filtered.empty:  # Handle empty data for the level
                    eval_scores[level] = 0.0
                    continue

                # Group by round_id to calculate score per object
                per_object_scores = filtered.groupby("round_id").apply(
                    lambda g: g['eval_score'].sum() / len(g) if g['label'].max() == 1 else 0
                )
                eval_scores[level] = per_object_scores.sum()

            # Calculate total
            if self.exc_grasps.empty:  # Handle empty data for the entire dataset
                eval_scores['total'] = 0.0
            else:
                per_object_scores_total = self.exc_grasps.groupby("round_id").apply(
                    lambda g: g['eval_score'].sum() / len(g) if g['label'].max() == 1 else 0
                )
                eval_scores['total'] = per_object_scores_total.sum()

            return eval_scores


def combine_trials(log_root_dir, expname):
    """Combine CSV files for all trials into a single set."""
    root_path = os.path.join(log_root_dir, "exp_results", expname)
    combined_path = root_path + "_combine"

    if not os.path.exists(combined_path):
        os.makedirs(combined_path)

    for file_type in ["exc_grasps", "pre_grasps", "rounds"]:
        df_combined = pd.DataFrame()
        round_dir_list = sorted(os.listdir(root_path))
        for i, round_dir in enumerate(round_dir_list):
            file_path = os.path.join(root_path, round_dir, f"{file_type}.csv")
            df_round = pd.read_csv(file_path)
            df_round["round_id"] = i
            df_combined = pd.concat([df_combined, df_round], ignore_index=True)
        df_combined.to_csv(os.path.join(combined_path, f"{file_type}.csv"), index=False)

    return Path(combined_path)


def main(args):
    gripper_type, expname = args.gripper_type, args.expname
    log_dir = (Path(args.log_root_dir) / 'exp_results'/ expname/ gripper_type).resolve()
    data = Data(log_dir)

    # Metrics computation
    print("Gripper:         ", Fore.GREEN, gripper_type, Fore.RESET)
    print("Experiment:      ", Fore.GREEN, expname, Fore.RESET)
    print("Num grasps:      ", Fore.GREEN, num_grasps := data.num_grasps(), Fore.RESET)
    print("Success rate:    ", Fore.GREEN, SR := data.success_rate(), Fore.RESET)
    print("Percent cleared: ", Fore.GREEN, DR := data.percent_cleared(), Fore.RESET)
    print("Eval score:      ", Fore.GREEN, ES := data.eval_score(), Fore.RESET)
    print("Planning time:  ", Fore.GREEN, IT := data.avg_planning_time(), Fore.RESET)


    # Save results
    if args.save:
        summary_file = os.path.join(log_dir,"summary_results.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Experiment name:    {expname}\n")
            f.write(f"Experiment name:    {expname}\n")
            f.write(f"Num grasps:         {num_grasps}\n")
            f.write(f"Success rate:       {SR}\n")
            f.write(f"Percent cleared:    {DR}\n")
            f.write(f"Eval score:         {ES}\n")
            f.write(f"Planning time:     {IT}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("log_root_dir", type=str, help="Root directory for logs")
    parser.add_argument("expname", type=str, help="Experiment name")
    parser.add_argument("gripper_type", type=str, help="Experiment name")
    parser.add_argument("save", type=int, help="Save results to file")
    args = parser.parse_args()
    main(args)
