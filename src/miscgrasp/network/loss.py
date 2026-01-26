import math
import sys

import mcubes
import numpy as np
import pyquaternion as pyq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from scipy.spatial.distance import cdist

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')

from src.miscgrasp.utils.base_utils import calc_rot_error_from_qxyzw_vgn, calc_rot_error_from_qxyzw_ga


class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys = keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass


class VGNADLoss(Loss):
    default_cfg = {
        'positive_w': 2.,
        'negative_w': 1.,
        'loss_vgn_weight': 1e-2,
    }

    def __init__(self, cfg):
        super().__init__(['loss_gavgn'])
        self.cfg = {**self.default_cfg, **cfg}

    def _loss_fn(self, y_pred, y, is_train):
        label_pred, rotation_pred, width_pred = y_pred
        label, rotation, width = y['label'], y['rotation'], y['width']

        loss_qual = self._qual_loss_fn(label_pred, label)
        acc = self._acc_fn(label_pred, label)

        loss_rot_raw = self._rot_loss_fn(rotation_pred, rotation)
        loss_rot = label * loss_rot_raw

        loss_width_raw = 0.01 * self._width_loss_fn(width_pred, width)
        loss_width = label * loss_width_raw
        loss = loss_qual + loss_rot + loss_width
        loss_item = {'loss_vgn': loss.mean()[None] * self.cfg['loss_vgn_weight'],
                     'vgn_total_loss': loss.mean()[None], 'vgn_qual_loss': loss_qual.mean()[None],
                     'vgn_rot_loss': loss_rot.mean()[None], 'vgn_width_loss': loss_width.mean()[None],
                     'vgn_qual_acc': acc[None]}

        if not is_train:
            num = torch.count_nonzero(label)
            angle_torch = label * self._angle_error_fn(rotation_pred.flatten(0, 1),
                                                       rotation.flatten(0, 1), 'torch').reshape(*rotation_pred.shape[:-1])
            loss_item['vgn_rot_err'] = (angle_torch.sum() / num)[None] if num else torch.zeros((1,), device=label.device)

        return loss_item

    def _qual_loss_fn(self, pred, target):
        weights = torch.where(target == 1,
                              target.new_full((), self.cfg['positive_w']),
                              target.new_full((), self.cfg['negative_w']),
                              )

        return F.binary_cross_entropy_with_logits(pred, target, weight=weights, reduction="none")

    def _acc_fn(self, pred, target):
        # NOTE: here the output quality is the logit, so it requires to be processed by sigmoid.
        return 100 * (torch.round(torch.sigmoid(pred)) == target).float().sum() / target.flatten(0, 1).shape[0]

    def _pr_fn(self, pred, target):
        p, r = torchmetrics.functional.precision_recall(torch.round(pred).to(torch.int), target.to(torch.int), 'macro',
                                                        num_classes=2)
        return p[None] * 100, r[None] * 100

    def _rot_loss_fn(self, pred, target):
        loss0 = self._quat_loss_fn(pred, target[..., 0, :])
        loss1 = self._quat_loss_fn(pred, target[..., 1, :])
        return torch.min(loss0, loss1)

    def _angle_error_fn(self, pred, target, method='torch'):
        if method == 'np':
            def _angle_error(q1, q2):
                q1 = pyq.Quaternion(q1[[3, 0, 1, 2]])
                q1 /= q1.norm
                q2 = pyq.Quaternion(q2[[3, 0, 1, 2]])
                q2 /= q2.norm
                qd = q1.conjugate * q2
                qdv = pyq.Quaternion(0, qd.x, qd.y, qd.z)
                err = 2 * math.atan2(qdv.norm, qd.w) / math.pi * 180
                return min(err, 360 - err)

            q1s = pred.detach().cpu().numpy()
            q2s = target.detach().cpu().numpy()
            err = []
            for q1, q2 in zip(q1s, q2s):
                err.append(min(_angle_error(q1, q2[0]), _angle_error(q1, q2[1])))
            return torch.tensor(err, device=pred.device)
        elif method == 'torch':
            return calc_rot_error_from_qxyzw_vgn(pred, target)
        else:
            raise NotImplementedError

    def _quat_loss_fn(self, pred, target):
        return 1.0 - torch.abs(torch.sum(pred * target, dim=-1))

    def _width_loss_fn(self, pred, target):
        return F.mse_loss(pred, target, reduction="none")

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        return self._loss_fn(data_pr['vgn_pred'], data_gt['grasp_info'], is_train)


class GraspSiamLoss(Loss):
    # Default configuration parameters
    default_cfg = {
        'loss_graspsiam_weight': 5e-3,  # Weight for the contrastive loss
    }

    def __init__(self, cfg):
        # Initialize the loss function with user-specified configuration
        super().__init__(['loss_siam'])
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        # Extract features for neural network, positive, and negative samples
        nn_fea, pos_fea = data_pr['nn_fea'], data_pr['pos_fea']

        # Normalize the features
        nn_fea_norm = F.normalize(nn_fea, dim=-1)
        pos_fea_norm = F.normalize(pos_fea, dim=-1)

        # Compute the similarity between positive features and neural network features
        loss = -torch.einsum('bic,bic->bi', pos_fea_norm, nn_fea_norm)

        loss_item = {'loss_siam': loss.mean()[None] * self.cfg['loss_graspsiam_weight']}
        return loss_item

    def mask_diag(self, similarity):
        # Mask diagonal elements in the similarity matrix (set them to negative infinity)
        B, I, J, _ = similarity.shape

        # Create a mask with ones on the diagonal and zeros elsewhere
        mask = torch.eye(J, J, device=similarity.device, dtype=torch.bool)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, J, J)
        mask = mask.expand(B, I, J, J)  # Shape: (B, I, J, J)

        # Apply the mask by setting diagonal elements to -inf
        similarity = similarity.masked_fill(mask, float('-inf'))

        return similarity


name2loss = {
    'vgn_adpt': VGNADLoss,
    'grasp_siam': GraspSiamLoss,
}