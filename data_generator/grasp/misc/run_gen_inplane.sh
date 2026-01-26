#!/bin/bash
IN_DIR=/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/train_raw_packed
OUT_DIR=/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/train_raw_packed

for i in {0..3}; do
  echo "Processing file $i..."
  /home/yons/anaconda3/envs/graspnerf/bin/python generate_inplane_label.py $i --in_dir $IN_DIR --out_dir $OUT_DIR
  if [ $? -eq 0 ]; then
    echo "File $i processed successfully."
  else
    echo "Error processing file $i. Exiting script."
    exit 1
  fi
done

echo "All files processed successfully!"


