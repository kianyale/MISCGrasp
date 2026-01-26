# import os
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
#
# # a = os.listdir('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data_generator/grasp/data/raw/foo/sensor_data')
# # b = np.random.choice(a)
# # c = np.load(os.path.join('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/datasets/data/GIGA/data_packed_train_raw/mesh_pose_list'
# #                          '/0a0be8d12a53475b93c4411a59cf5279.npz'),
# #             allow_pickle=True)['pc'][0]
# # print(c.dtype)
# # print(c[0].dtype, c[1].dtype, c)
# c = [['/dad/dsadsa/dsd', 0.88, np.array([1, 2, 3])],
#      ['/dad/dsadsa/dsd', 0.88, np.array([1, 2, 3])]]
# # c = ['/dad/dsadsa/dsd', 0.88, np.array([1, 2, 3])]
# c = np.asarray(c, dtype=object)
# print(c)
# np.savez_compressed('~/temp/temp.npz', pc=c)
import math
import time

# import numpy as np
# from scipy.interpolate import interpn
#
# # 定义网格点
# x = np.linspace(0, 4, 5)  # x 轴有 5 个采样点
# y = np.linspace(0, 4, 5)  # y 轴有 5 个采样点
# z = np.linspace(0, 4, 5)  # z 轴有 5 个采样点
#
# # 网格点上给定的值，形状为 (5, 5, 5)
# values = np.random.rand(5, 5, 5)
#
# # 定义 points，包含每个维度的采样点
# points = (x, y, z)
#
# # 定义要插值的目标点
# xi = np.array([[2.1, 2.1, 2.1], [1.5, 1.5, 1.5]])  # 插值点
# yi = np.array([1.5, 1.5, 1.5])  # 插值点
#
# a = [np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1])]
# print(np.asarray(a).shape)
# # 进行三线性插值
# result = interpn(points, values, xi, method='linear')
# print(result.shape)
# print("Interpolated values at points:", result)

import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.use('QT5Agg')

structure = np.ones([1, 3, 3, 3], dtype=bool)
structure[:, :, :, 0:1] = False
print(structure)


def regularize_tsdf_grid(tsdf_grid, threshold=0.9):
    """
    对 TSDF 网格中的体素进行扩张操作，直到外部截断的体素都变为1。

    Args:
        tsdf_grid (numpy.ndarray): 形状为 (1, res_x, res_y, res_z) 的 TSDF 网格
        threshold (float): 值大于该阈值的体素将作为扩张种子

    Returns:
        numpy.ndarray: 扩张后的 TSDF 网格
    """
    # 获取 TSDF grid 的形状
    grid_shape = tsdf_grid.shape

    # 找到值大于阈值（0.9）的体素，作为种子区域
    seed_mask = tsdf_grid > threshold
    protect_mask = tsdf_grid == 0.

    # 将膨胀区域的值设为 True，截断的体素 (值为 0) 将被扩展到 1
    dilated_mask = seed_mask.copy()  # 初始化膨胀区域
    new_dilated_mask = binary_dilation(dilated_mask, structure=structure, iterations=0, mask=protect_mask)

    tsdf_grid[new_dilated_mask & (tsdf_grid == 0)] = 1.

    return tsdf_grid


def vis(g):
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.voxels(np.logical_and(g < 0.5, g < 0.5), edgecolor='k')
    ax.set_xlabel('X label')
    ax.set_xlabel('Y label')
    ax.set_xlabel('Z label')

    plt.show()


if __name__ == '__main__':
    # 示例使用：
    # 假设我们有一个 TSDF grid，值域为 [0, 1)，形状为 (1, 160, 160, 160)
    # tsdf_grid = np.load('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/tsdf_grid.npy')
    #
    # t0 = time.time()
    # # 对大于0.9的体素进行扩张
    # expanded_grid = regularize_tsdf_grid(tsdf_grid, threshold=0.9)
    # t1 = time.time()
    #
    # print(t1 - t0)
    #
    # vis(expanded_grid[0])
    a = 2
    if 1 < a < 3:
        print('tr')
    print(math.atan(1))