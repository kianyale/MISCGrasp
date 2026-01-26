#!/bin/bash
N_ROUND=10000
mycnt=546 # start from 0
seed=2185 # start from 1
#N_ROUND=20000
#mycnt=10222 # start from 0
#seed=10889 # start from 1

while (( $mycnt < $N_ROUND ))
do
  printf "Process %s is starting: \n\n" "$(( mycnt+1 ))"
#  mpirun -np 4 /home/yons/anaconda3/envs/graspnerf/bin/python generate_data_precompute.py \
  /home/yons/anaconda3/envs/graspnerf/bin/python generate_data_precompute.py \
  /media/yons/6d379145-c1d8-430f-9056-7777219c83a8/temp \
  --seed $seed --scene pile --object-set egad --sim-gui
  printf "Process %s is done: \n\n" "$(( mycnt+1 ))"
  if (( ($mycnt + 1) % 10 == 0))
  then
    printf "Cleanup %s is starting: \n" "$(( ($mycnt+1) % 10 ))"
#    /home/yons/anaconda3/envs/graspnerf/bin/python cleanup.py /media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/train_raw_pile --flag train
    printf "Cleanup %s is done: \n\n" "$(( ($mycnt+1) % 10 ))"
  fi
((mycnt=$mycnt+1));
((seed=$seed+4));
done;