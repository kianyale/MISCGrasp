#!/bin/bash
N_ROUND=100
mycnt=0 # start from 0
scene=packed
object_set=packed
object_scale=small

while (( $mycnt < $N_ROUND ))
do
  printf "Process %s is starting: \n\n" "$(( mycnt+1 ))"
  /home/yons/anaconda3/envs/graspnerf/bin/python generate_test_scene.py \
  /home/yons/temp/pinch_packed_new \
  --seed $mycnt --scene $scene --object-set $object_set --object-scale $object_scale --rounds $N_ROUND #--sim-gui
  printf "Process %s is done: \n\n" "$(( mycnt+1 ))"
#  if (( ($mycnt + 1) % 5 == 0))
#  then
#    printf "Cleanup %s is starting: \n" "$(( ($mycnt+1) % 5 ))"
#    /home/yons/anaconda3/envs/graspnerf/bin/python cleanup.py \
#    "/home/yons/temp/temp/grasps/${scene}_${object_set}_${object_scale}_${N_ROUND}" --flag "test"
#    printf "Cleanup %s is done: \n\n" "$(( ($mycnt+1) % 5 ))"
#  fi
((mycnt=$mycnt+1));
done