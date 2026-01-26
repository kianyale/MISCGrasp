import collections
import time
import uuid
import os

import mcubes
import numpy as np
import pandas as pd
import open3d as o3d
import sys

import yaml
from colorama import *

sys.path.append("/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp")
import shutil
from pathlib import Path
from src.gd import io, vis
from src.gd.grasp import *
from src.gd.simulation import ClutterRemovalSim
from src.gd.utils.transform import Rotation, Transform


Idx2Name = [f"{chr(65 + i // 7)}{i % 7}" for i in range(49)]

two_finger = [
    'franka',
    'robotiq_2f_85',
    'robotiq_2f_140',
    'wsg_32',
    'ezgripper',
    'sawyer',
    'wsg_50',
    'rg2',
    'barrett_hand_2f',
]
three_finger = [
    'kinova_3f',
    'robotiq_3f',
    'barrett_hand'
]

State = collections.namedtuple("State", ["tsdf", "pc"])


def copydirs(from_file, to_file):
    if not os.path.exists(to_file):
        os.makedirs(to_file)
    files = os.listdir(from_file)
    for f in files:
        if os.path.isdir(from_file + '/' + f):
            copydirs(from_file + '/' + f, to_file + '/' + f)
        else:
            shutil.copy(from_file + '/' + f, to_file + '/' + f)


def run(
        grasp_plan_fn,
        logdir,
        description,
        scene,
        object_set,
        num_objects=5,
        n=6,
        N=None,
        seed=1,
        sim_gui=False,
        rviz=False,
        round_idx=0,
        asset_dir="",
        args=None,
        scene_o3d_vis=False,
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, renderer_root_dir=asset_dir, args=args)
    sim.change_gripper(args.gripper_type, args.gripper_scale)
    pre_depth = sim.gripper.finger_depth_init if hasattr(sim.gripper, 'finger_depth_init') else sim.gripper.finger_depth
    filter_depth = pre_depth if pre_depth >= sim.gripper.finger_depth else sim.gripper.finger_depth

    if scene == 'single':
        logger = LoggerSingle(asset_dir, args.log_root_dir, logdir, round_idx, args.gripper_type, args.obj_scale)
    elif scene == 'pile' or scene == 'packed':
        logger = LoggerMulti(asset_dir, args.log_root_dir, logdir, round_idx, args.gripper_type, args.obj_scale)
    else:
        raise NotImplementedError

    for n_round in range(round_idx, round_idx + 1):
        urdfs_and_poses_dict = sim.reset(num_objects, round_idx)

        logger.log_round(round_idx, sim.num_objects)

        consecutive_failures = 0
        last_label = None

        n_grasp = 0
        while sim.num_objects > 0:

            timings = {}

            gt_tsdf, gt_tsdf_hr, timings['integration'] = sim.acquire_tsdf(n=n, N=N, suffix=round_idx)  # tsdf in here ranges from 0 to 1
            pc = gt_tsdf_hr.get_cloud()
            # pc_temp = np.asarray(pc.points).copy()
            # tsdf_temp = gt_tsdf.get_grid().squeeze()
            # np.savez_compressed('temp', pc=pc_temp, tsdf=tsdf_temp)

            if scene_o3d_vis:
                o3d.visualization.draw_geometries([gt_tsdf.get_cloud()])
                o3d.visualization.draw_geometries([gt_tsdf_hr.get_cloud()])

            if rviz:
                vis.clear()
                vis.draw_workspace(sim.size)
                vis.draw_tsdf(gt_tsdf_hr.get_grid().squeeze(), gt_tsdf_hr.voxel_size)
                vis.draw_points(np.asarray(pc.points))

            state = State(gt_tsdf, pc)
            if args.method == "vgn":
                grasps, scores, timings["planning"] = grasp_plan_fn(round_idx, n_grasp, state,
                                                                    args.gripper_type,
                                                                    args.gripper_scale,
                                                                    filter_depth,
                                                                    args.choose_best)
            elif args.method == "ga_vgn":
                grasps, scores, timings["planning"] = grasp_plan_fn(round_idx, n_grasp, state,
                                                                    args.gripper_type,
                                                                    args.gripper_scale,
                                                                    filter_depth,
                                                                    args.choose_best)
            elif args.method == "giga":
                grasps, scores, timings["planning"] = grasp_plan_fn(round_idx, n_grasp, state,
                                                                    args.gripper_type,
                                                                    args.gripper_scale,
                                                                    filter_depth,
                                                                    args.choose_best)
            else:
                raise NotImplementedError

            if len(grasps) == 0:
                print(f"[I] {Fore.YELLOW}No detections found, abort this round{Fore.RESET}")
                break
            else:
                print(f"[I] {Fore.GREEN}{len(grasps)}{Fore.RESET} detections are found")

            # execute grasp
            grasp, score = grasps[0], scores[0]  # selection from random permutation
            if rviz and args.gripper_type in two_finger:
                real_width_grasps = []
                for g in grasps:
                    # g.width = sim.gripper.max_opening_width
                    real_width_grasps.append(g)
                vis.draw_grasp(real_width_grasps[0], score, pre_depth)
                vis.draw_grasps(real_width_grasps, scores, pre_depth)
            elif rviz and args.gripper_type in three_finger:
                w = (sim.gripper.finger_open_distance_right, sim.gripper.finger_open_distance_left)
                bias = sim.gripper.half_gap_2f
                vis.draw_grasp_3f(grasp, score, pre_depth, bias, w)
                vis.draw_grasps_3f(grasps, scores, pre_depth, bias, w)

            if rviz and input(
                    f'If you\'d like to execute, input `{Fore.GREEN}y{Fore.RESET}` and press `{Fore.GREEN}Enter{Fore.RESET}`: ') == 'y':
                (label, _), remain_obj_inws_infos = sim.execute_grasp(grasp, allow_contact=True, finger_depth=pre_depth)
                time.sleep(1)
            else:
                (label, _), remain_obj_inws_infos = sim.execute_grasp(grasp, allow_contact=True, finger_depth=pre_depth)

            # log the grasp
            logger.log_grasp(round_idx, state, timings, grasp, score, label)
            if scene == 'single':
                logger.log_grasps(round_idx, timings, grasps, scores)

            # Increment or reset the failure counter
            if label == Label.FAILURE:
                consecutive_failures += 1
                print(f"[I] {Fore.YELLOW}Grasp failed! Consecutive failures: {consecutive_failures}{Fore.RESET}")
            else:
                consecutive_failures = 0

            # Check if the maximum allowed consecutive failures is reached
            if consecutive_failures >= args.max_consecutive_fail:
                print(f"[I] {Fore.YELLOW}Reached {consecutive_failures} consecutive failures. Aborting...{Fore.RESET}")
                break

            # If there are no objects left, break out of the loop as well
            if sim.num_objects <= 0:
                break

            n_grasp += 1


class LoggerSingle(object):
    def __init__(self, asset_dit, log_root_dir, expname, round_idx, gripper_type, obj_scale):
        score_path = os.path.join(asset_dit, 'score', f'{gripper_type}.yaml')
        with open(score_path, 'r') as f:
            self.score = yaml.full_load(f)
        self.logdir = Path(log_root_dir) / "exp_results" / expname / gripper_type

        self.obj_scale = obj_scale
        if obj_scale == 'small':
            self.round_idx = round_idx
        elif obj_scale == 'medium':
            self.round_idx = round_idx + 49
        elif obj_scale == 'large':
            self.round_idx = round_idx + 49 * 2
        else:
            raise NotImplementedError

        self.scenes_dir = self.logdir / "sensor_data"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.pre_grasps_csv_path = self.logdir / "pre_grasps.csv"
        self.exc_grasps_csv_path = self.logdir / "exc_grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id",
                                                 "scene_id",
                                                 "object_scale",
                                                 "level",
                                                 "object_count"])

        if not self.exc_grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "object_scale",
                "level",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "planning_time",
                "eval_score"
            ]
            io.create_csv(self.exc_grasps_csv_path, columns)

        if not self.pre_grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "object_scale",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "planning_time"
            ]
            io.create_csv(self.pre_grasps_csv_path, columns)

    def log_round(self, round_id, object_count):
        scene_id = Idx2Name[round_id]
        io.append_csv(self.rounds_csv_path,
                      self.round_idx,
                      scene_id,
                      self.obj_scale,
                      self.score[self.obj_scale][scene_id]['level'],
                      object_count)

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        scene_id = Idx2Name[round_id]

        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_path = self.scenes_dir / f"{scene_id}_{self.obj_scale}.npz"
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        eval_score = self.score[self.obj_scale][scene_id]['final_score'] if label else 0.
        level = self.score[self.obj_scale][scene_id]['level']
        io.append_csv(
            self.exc_grasps_csv_path,
            self.round_idx,
            scene_id,
            self.obj_scale,
            level,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["planning"],
            eval_score
        )

    def log_grasps(self, round_id, timings, grasps, scores):
        for grasp, score in zip(grasps, scores):
            scene_id = Idx2Name[round_id]

            # log grasp
            qx, qy, qz, qw = grasp.pose.rotation.as_quat()
            x, y, z = grasp.pose.translation
            width = grasp.width
            io.append_csv(
                self.pre_grasps_csv_path,
                self.round_idx,
                scene_id,
                self.obj_scale,
                qx,
                qy,
                qz,
                qw,
                x,
                y,
                z,
                width,
                score,
                timings["planning"],
            )


class LoggerMulti(object):
    def __init__(self, asset_dit, log_root_dir, expname, round_idx, gripper_type, obj_scale):
        self.logdir = Path(log_root_dir) / "exp_results" / expname / ("%04d" % int(round_idx))
        self.scenes_dir = self.logdir / "sensor_data"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


def compute_mae(pr, gt, mask):
    return np.mean(np.abs(pr[mask] - gt[mask]))


class Data(object):
    """
    Object for loading and analyzing experimental data.
    """

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
        scene_id, gripper_type, scale, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "sensor_data" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
