import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import rospy

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')

from src.miscgrasp.main import VGNPlanner
from src.gd.experiments import clutter_removal


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    if args.method == "vgn":
        grasp_planner = VGNPlanner(args)
    else:
        raise NotImplementedError("No such method!")

    if args.rviz:
        rospy.init_node("vgn_vis", anonymous=True)

    clutter_removal.run(
        grasp_plan_fn=grasp_planner,
        logdir=args.logdir,
        description=args.description,
        scene=args.scene,
        object_set=args.object_set,
        num_objects=args.num_objects,
        seed=args.seed,
        sim_gui=args.sim_gui,
        rviz=args.rviz,
        round_idx=args.round_idx,
        asset_dir=args.asset_dir,
        args=args,
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = argparse.ArgumentParser()

        # 手动解析的参数作为命令行参数
        parser.add_argument("round_idx", type=int, help="Round index")
        parser.add_argument("gpuid", type=int, help="GPU ID")
        parser.add_argument("logdir", type=str, help="Experiment name")
        parser.add_argument("scene", type=str, choices=["pile", "packed", "single"], help="Scene type")
        parser.add_argument("object_set", type=str, help="Object set")
        parser.add_argument("check_seen_scene", type=int, help="Seen scene flag (0/1)")
        parser.add_argument("asset_dir", type=str, help="Path to asset directory")
        parser.add_argument("log_root_dir", type=str, help="Log root directory")
        parser.add_argument("method", type=str, help="Method")
        parser.add_argument("gripper_type", type=str, help="Gripper type")
        parser.add_argument("gripper_scale", type=float, help="Gripper scale")
        parser.add_argument("obj_scale", type=str, help="Object scale")
        parser.add_argument("sim_gui", type=int, help="Enable GUI (0/1)")
        parser.add_argument("rviz", type=int, help="Enable RViz (0/1)")
        parser.add_argument("choose_best", type=str, help="Enable Choose Best/Random/Highest")
        parser.add_argument("max_consecutive_fail", type=int, help="Maximum number of consecutive failures allowed")

        # 其他 argparse 参数
        parser.add_argument("--model", type=Path, default="")
        parser.add_argument("--description", type=str, default="")
        parser.add_argument("--num-objects", type=int, default=10)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--cfg_fn", type=str, default="src/miscgrasp/configs/gavgn.yaml")
        parser.add_argument("--gen_scene_descriptor", action="store_true", default=False)
        parser.add_argument("--gen_test_scene_descriptor", action="store_true", default=False)
        parser.add_argument("--load_scene_descriptor", action="store_true", default=True)
        parser.add_argument("--camera_focal", type=float, default=459.14)

        args = parser.parse_args()

        # 调整布尔值
        args.sim_ = bool(args.sim_gui)
        args.rviz = bool(args.rviz)

        print("########## Simulation Start ##########")
        print(f"Object Set: {args.object_set}\nScene: {args.scene}\nRound: {args.round_idx}\nMethod: {args.method}")
        print("######################################")

        main(args)
