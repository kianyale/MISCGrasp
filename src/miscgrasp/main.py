import sys
import time
from math import floor

sys.path.append("./src/miscgrasp")
from pathlib import Path
import numpy as np

import torch
from utils.base_utils import load_cfg, to_cuda
from utils.imgs_info import grasp_info_to_torch
from network.wrapper import name2network
from network.loss import VGNADLoss
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from colorama import *

from src.gd.utils.transform import Transform, Rotation
from src.gd.grasp import *
from src.gd.perception import TSDFVolume
from src.gd import vis


def process(
        tsdf_vol,
        qual_vol,
        rot_vol,
        width_vol,
        gaussian_filter_sigma=1.0,
        min_width=0.4,  # TODO
        max_width=15.6,  # TODO
        tsdf_thres_high=0.5,
        tsdf_thres_low=1e-3,
        iter=None
):
    tsdf_vol = tsdf_vol.squeeze()
    qual_vol = qual_vol.squeeze()
    rot_vol = rot_vol.squeeze()
    width_vol = width_vol.squeeze()
    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > tsdf_thres_high  # NOTE: it is equal to `tsdf_thres_high < tsdf_vol < 1`
    inside_voxels = np.logical_and(tsdf_thres_low < tsdf_vol, tsdf_vol < tsdf_thres_high)
    if iter is not None:
        valid_voxels = ndimage.morphology.binary_dilation(
            outside_voxels, iterations=iter, mask=np.logical_not(inside_voxels)
        )
    else:
        valid_voxels = ndimage.morphology.binary_dilation(
            outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
        )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.9, max_filter_size=8):
    # vis(qual_vol, 1)
    qual_vol[qual_vol < threshold] = 0.0
    # vis(qual_vol, 2)

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)

    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)
    # vis(qual_vol, 3)

    # construct grasps
    grasps, scores, indexs = [], [], []

    for index in np.argwhere(mask):
        indexs.append(index)
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)
    return grasps, scores, indexs


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    rot = rot_vol[:, i, j, k]
    ori = Rotation.from_quat(rot)
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score


class VGNPlanner(object):
    def set_params(self, args):
        self.args = args
        self.trunc = 8
        self.bbox3d = [[-0.2, -0.2, -0.05], [0.2, 0.2, 0.2497]]
        self.tsdf_thres_high = 0
        self.tsdf_thres_low = -0.95

    def __init__(self, args=None, cfg_fn=None, debug_dir=None) -> None:
        # load render cfg
        if cfg_fn is None:
            self.set_params(args)
            cfg = load_cfg(args.cfg_fn)
        else:
            cfg = load_cfg(cfg_fn)

        print(f"[I] {Fore.GREEN}GAVGNPlanner{Fore.RESET}: using ckpt: {Fore.GREEN}{cfg['name']}{Fore.RESET}")
        # load model
        self.net = name2network[cfg['network']](cfg)
        ckpt_filename = 'best_model_174900'  # NOTE
        path = Path('src/miscgrasp/data/model') / cfg["group_name"] / cfg["name"] / f'{ckpt_filename}.pth'
        ckpt = torch.load(
            Path('src/miscgrasp/data/model') / cfg["group_name"] / cfg["name"] / f'{ckpt_filename}.pth')  # NOTE
        self.net.load_state_dict(ckpt['network_state_dict'])
        self.net.cuda()
        self.net.eval()
        self.rviz = args.rviz
        self.step = ckpt["step"]
        self.output_dir = debug_dir
        if debug_dir is not None:
            if not Path(debug_dir).exists():
                Path(debug_dir).mkdir(parents=True)
        self.loss = VGNADLoss({})
        print(f"[I] {Fore.GREEN}GAVGNPlanner{Fore.RESET}: load model at step {Fore.GREEN}{self.step}{Fore.RESET} of"
              f" best metric {Fore.GREEN}{ckpt['best_para']}{Fore.RESET}")

    def __call__(self, round_idx, n_grasp, state, gripper_type, gripper_scale, filter_depth, choose):
        """
        Parameters:
            round_idx (int): the index of the current round.
            n_grasp (int): the index of the current grasp attempt in one round.
            tsdf (TSDFVolume): planning time
        """
        # load data for test
        scene_grid = state.tsdf.get_grid() * 2 - 1
        scene_grid = scene_grid.squeeze()
        voxel_size = state.tsdf.voxel_size

        qual_vol_ori, rot_vol_ori, width_vol_ori, toc = self.core(scene_grid)

        iter = floor(filter_depth / voxel_size) - 8
        qual_vol, rot_vol, width_vol = process(scene_grid, qual_vol_ori, rot_vol_ori, width_vol_ori,
                                               tsdf_thres_high=self.tsdf_thres_high, tsdf_thres_low=self.tsdf_thres_low,
                                               iter=iter)
        grasps, scores, indexs = select(qual_vol.copy(), rot_vol, width_vol)
        grasps, scores, indexs = np.asarray(grasps), np.asarray(scores), np.asarray(indexs)

        if len(grasps) > 0:
            np.random.seed(self.args.seed + round_idx + n_grasp)
            if choose == 'best':
                p = np.argsort(-scores)  # 按分数排序
            elif choose == 'random':
                p = np.random.permutation(len(grasps))  # 随机排列
            elif choose =='highest':
                p = np.argsort([-g.pose.translation[2] for g in grasps])
            else:
                raise NotImplementedError
            # 重排 grasps 列表
            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
            # 重排 scores 和 indexs
            scores = scores[p]
            indexs = indexs[p]

        if self.rviz:
            vis.draw_quality(qual_vol, voxel_size, threshold=0.)

        return grasps, scores, toc

    def core(self, scene_tsdf, gt_info=None):
        """
        Parameters:
            tsdf_vol (np.ndarray): np array of shape (res, res, res), input TSDF
            gt_info (dict): ground truth info

        Returns:
            tuple: A tuple containing the following elements:
                    - label (np.ndarray): np array of shape (1, 1, res, res, res)
                    - rot (np.ndarray): np array of shape (1, 4, res, res, res)
                    - width (np.ndarray): np array of shape (1, 1, res, res, res)
                    - t (float): planning time
            done
        """
        data = {}
        data = {'step': self.step, 'eval': True}
        data['geo_info'] = {'scene': torch.from_numpy(scene_tsdf).float()[None, None, ...]}
        if not gt_info:
            data['full_vol'] = True
        else:
            data['grasp_info'] = to_cuda(grasp_info_to_torch(gt_info))
        data['geo_info'] = to_cuda(data['geo_info'])

        with torch.no_grad():
            t0 = time.time()
            outputs = self.net(data)
            t = time.time() - t0

        if gt_info:
            return self.loss(outputs, data, self.step, False)

        label, rot, width = outputs['vgn_pred']
        label = torch.sigmoid(label)
        label, rot, width = label.squeeze(1), rot.squeeze(1), width.squeeze(1)

        return label.cpu().numpy(), rot.cpu().numpy(), width.cpu().numpy(), t


def vlz(g, i):
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.voxels(np.logical_and(g > 0.5, g < 1), edgecolor='k')
    ax.set_xlabel('X label')
    ax.set_xlabel('Y label')
    ax.set_xlabel('Z label')

    plt.savefig(f'/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/img{i}.png')