import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


mpl.use('TKAgg')

def paint(g):
    fig = plt.figure() # 创建一个画布figure，然后在这个画布上加各种元素。
    ax = Axes3D(fig) # 将画布作用于 Axes3D 对象上
    # ax = plt.axes(projection='3d')
    fig.add_axes(ax)
    ax.voxels(np.logical_and(g < 0, g < 0), edgecolor='k')
    x_max, y_max, z_max = g.shape
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, y_max])
    ax.set_zlim([0, z_max])
    ax.set_xlabel('X label') # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')

    plt.show()

gripper_types = [
  'franka',
  'robotiq_2f_85',
  'robotiq_2f_140',
  'wsg_32',
  'ezgripper',
  'sawyer',
  'wsg_50',
  'rg2',
  'barrett_hand_2f',
  'kinova_3f',
  'robotiq_3f',
  'barrett_hand'
]

if __name__ == '__main__':
    for grp in gripper_types:
        tsdf = np.load(f'/home/yons/temp2/{grp}_1.0.npz')
        paint(tsdf['close'].squeeze())