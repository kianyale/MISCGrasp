import pandas as pd

file_path = '/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/train_raw_packed/grasps.csv'
out_path = '/grasps.csv'

df = pd.read_csv(file_path, on_bad_lines='skip')

df.to_csv(out_path, index=False)
print('Successfully generated csv file in ', out_path)