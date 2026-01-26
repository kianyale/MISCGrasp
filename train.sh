cd src/miscgrasp || exit
CUDA_VISIBLE_DEVICES=$1 ~/anaconda3/envs/graspnerf/bin/python run_training.py \
                        --cfg configs/miscgrasp.yaml
cd - || exit