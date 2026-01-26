import pandas as pd

# 读取两个 CSV 文件
file1 = "/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_vgn/washed/packed/grasps.csv"
file2 = "/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_vgn/washed/pile/grasps.csv"
# file3 = "/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/train_raw_pile_part3/grasps.csv"
out = "/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/vlz/temp/grasps_vgn.csv"

df1 = pd.read_csv(file1)
df1 = df1[df1['gripper_type'] == 'franka']
df2 = pd.read_csv(file2)
df2 = df2[df2['gripper_type'] == 'franka']

# df3 = pd.read_csv(file3)

# 按行拼接（默认）
df_combined_row = pd.concat([df1, df2], axis=0)

# 保存结果到新文件
df_combined_row.to_csv(out, index=False)

print("拼接完成！")
