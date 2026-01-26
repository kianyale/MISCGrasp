#!/bin/bash

GPUID=0

ASSET_DIR=./data/assets
SIM_LOG_DIR="./log/`date '+%Y%m%d-%H%M%S'`"
#SIM_LOG_DIR="./log/20250105-143432"

scene="single"
object_set="egad"
expname="test_single"

check_seen_scene=0
NUM_TRIALS=49
METHOD='vgn'  # ['vgn', 'giga']
CFG_FN="src/miscgrasp/configs/gavgn.yaml"

gripper_scale=1
GRIPPER_TYPES=(
    'franka'
)
OBJ_SCALES=('small' 'medium' 'large')

GUI=0
RVIZ=0
CHOOSE="best"  # [best, random, highest]
MAX_CONSECUTIVE_FAILURES=1

for gripper in "${GRIPPER_TYPES[@]}"
do
  for obj_scale in "${OBJ_SCALES[@]}"
  do
    mycount=0
    while (( $mycount < $NUM_TRIALS ))
    do
       /home/yons/anaconda3/envs/graspnerf/bin/python scripts/sim_grasp.py \
       $mycount $GPUID $expname $scene $object_set $check_seen_scene  \
       $ASSET_DIR $SIM_LOG_DIR $METHOD $gripper $gripper_scale $obj_scale \
       $GUI $RVIZ $CHOOSE $MAX_CONSECUTIVE_FAILURES --load_scene_descriptor --cfg_fn $CFG_FN

       /home/yons/anaconda3/envs/graspnerf/bin/python ./scripts/single_stats_summary.py $SIM_LOG_DIR \
          $expname $gripper 0
    ((mycount=$mycount+1))
    done
  done
  /home/yons/anaconda3/envs/graspnerf/bin/python ./scripts/single_stats_summary.py $SIM_LOG_DIR \
    $expname $gripper 1
done
