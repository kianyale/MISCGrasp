#!/bin/bash

GRIPPER_TYPES=(
    'franka'
    'robotiq_2f_85'
    'robotiq_2f_140'
    'wsg_32'
    'ezgripper'
    'sawyer'
    'wsg_50'
    'rg2'
    'barrett_hand_2f'
    'kinova_3f'
    'robotiq_3f'
    'barrett_hand'
#    'leap_hand_right',
    'h5_hand'
)

OBJ_SCALES=('small') # 'medium' 'large')

for obj_scale in "${OBJ_SCALES[@]}"
do
  for gripper in "${GRIPPER_TYPES[@]}"
  do
    /home/yons/anaconda3/envs/graspnerf/bin/python obj_evaluate.py \
      /media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/score_raw --gripper "$gripper" --object-scale "$obj_scale" --interval 5 #--sim-gui
  done
done
