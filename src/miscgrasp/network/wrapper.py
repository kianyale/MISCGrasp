import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.modules import (
    NNCLRPredictionHead,
    NNMemoryBankModule,
    SimSiamProjectionHead,
)

from src.gd.networks import get_network


class MISCWrapper(nn.Module):
    base_cfg = {

    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.base_cfg, **cfg}
        self.vgn_net = get_network(self.cfg['backbone'])({})
        if self.cfg['backbone'] in self.cfg['multi_scale']:
            self.projection_head = SimSiamProjectionHead(224,
                                                         128,
                                                         128)  # TODO: Implement projection head
            self.multi_scale = True
            self.memory_banks = self._initialize_memory_banks()


        else:
            self.projection_head = SimSiamProjectionHead(128,
                                                         96,
                                                         96)  # TODO: Implement projection head
            self.multi_scale = False
            self.memory_banks = self._initialize_memory_banks()


    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data (dict): Contains 'geo_info' (scene and gripper grids)
                         and 'grasp_info' (grasp positions, indices, labels, etc.).

        Returns:
            dict: Contains 'vgn_pred' (grasp quality, rotation, width)
                  and memory features for positive/negative examples.
        """
        scene_grid = data['geo_info']['scene']
        vgn_outputs = self.vgn_net(scene_grid)

        if 'full_vol' not in data:
            is_train = data['is_train']

            index = data['grasp_info']['index']

            outputs = {'vgn_pred': self.select(vgn_outputs, index)}

            pos = data['grasp_info']['pos']
            if self.cfg['use_projection']:
                proj_fea = self.trilinear_interpolate(vgn_outputs, pos)  #  positive/negative features are mixed
                outputs['nn_fea'], outputs['pos_fea'], outputs['neg_fea'] = self.query_memory_banks(proj_fea,
                                                                                                    data['grasp_info'][
                                                                                                        'label'],
                                                                                                    is_train)
            else:
                inter_fea = self.trilinear_interpolate(vgn_outputs, pos)
                outputs['nn_fea'], outputs['pos_fea'], outputs['neg_fea'] = self.query_memory_banks(inter_fea,
                                                                                                    data['grasp_info'][
                                                                                                        'label'],
                                                                                                    is_train)
        else:
            outputs = {'vgn_pred': (vgn_outputs['qual'], vgn_outputs['rot'], vgn_outputs['width'])}

        return outputs

    def _initialize_memory_banks(self):
        """Initialize memory banks for each gripper type."""
        if self.multi_scale:
            return nn.ModuleDict({
                grp: NNMemoryBankModule(size=(1024 * 32, 128)) for grp in self.cfg['gripper_types']  # 128
            })
        else:
            return nn.ModuleDict({
                grp: NNMemoryBankModule(size=(1024 * 32, 96)) for grp in self.cfg['gripper_types']
            })

    def select(self, out, index):
        """Select specific grasp predictions from the VGN output."""
        qual_out, rot_out, width_out = out['qual'], out['rot'], out[
            'width']  # (2, 1, 80, 80, 80), (2, 4, 80, 80, 80), (2, 1, 80, 80, 80)
        b_scn, num_grasp, _ = index.shape  # index.shape = (2, 24, 3)

        # Flatten index for advanced indexing
        index_flat = index  # shape (B, 24, 3)
        scn_index = torch.arange(b_scn, device=index.device).view(-1, 1)  # shape (B, 1)

        # Select grasp qualities
        label = qual_out[scn_index, :, index[..., 0], index[..., 1], index[..., 2]].squeeze(-1)  # Shape: (B, 24)

        # Select grasp rotations
        rot = rot_out[scn_index, :, index[..., 0], index[..., 1], index[..., 2]]  # Shape: (B, 24, 4)

        # Select grasp widths
        width = width_out[scn_index, :, index[..., 0], index[..., 1], index[..., 2]].squeeze(-1)  # Shape: (B, 24)

        return label, rot, width

    def trilinear_interpolate(self, out, pos):
        """Perform trilinear interpolation on the output features."""
        b_scn, num_grasp, _ = pos.shape
        norm_pos = 2 * pos / self.cfg['volume_res'] - 1  # Normalize grasp positions
        if not self.multi_scale:
            # Reshape and interpolate
            agg_fea = out['agg_fea']  # (B, 128, 5, 5, 5)
            norm_pos_reshape = norm_pos[:, :, None, None, :]
            inter_fea = F.grid_sample(agg_fea, norm_pos_reshape, mode='bilinear', padding_mode='zeros',
                                      align_corners=True)

            inter_fea_reshape = inter_fea.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(b_scn, num_grasp, -1)

        else:
            # Reshape and interpolate
            agg_fea1, agg_fea2, agg_fea3 = out['agg_fea1'], out['agg_fea2'], out['agg_fea3']  # (B, 128, 5, 5, 5)
            norm_pos_reshape = norm_pos[:, :, None, None, :]
            inter_fea1 = F.grid_sample(agg_fea1, norm_pos_reshape, mode='bilinear', padding_mode='zeros',
                                      align_corners=True)
            inter_fea2 = F.grid_sample(agg_fea2, norm_pos_reshape, mode='bilinear', padding_mode='zeros',
                                      align_corners=True)
            inter_fea3 = F.grid_sample(agg_fea3, norm_pos_reshape, mode='bilinear', padding_mode='zeros',
                                      align_corners=True)

            # Reshape interpolated features
            inter_fea_reshape1 = inter_fea1.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(b_scn, num_grasp, -1)
            inter_fea_reshape2 = inter_fea2.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(b_scn, num_grasp, -1)
            inter_fea_reshape3 = inter_fea3.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(b_scn, num_grasp, -1)
            inter_fea_reshape = torch.cat([inter_fea_reshape1, inter_fea_reshape2, inter_fea_reshape3], dim=-1)

        if self.cfg['use_projection']:
            proj_fea = self.projection_head(inter_fea_reshape.flatten(0, 1)).reshape(b_scn, num_grasp, -1)
            return proj_fea

        return inter_fea_reshape

    def query_memory_banks(self, fea, label, is_train):
        """Query the memory bank for positive and negative features."""
        b_scn, num_grasp, d_fea = fea.shape
        label = label.squeeze(-1)
        positive_mask = label == 1
        negative_mask = label == 0

        # Extract positive and negative features
        positive_fea = fea[positive_mask].reshape(b_scn, -1, d_fea)
        negative_fea = fea[negative_mask].reshape(b_scn, -1, d_fea)
        num_pos_grasp = positive_fea.shape[-2]

        # Efficient memory bank query
        fea_ = positive_fea.flatten(0, 1)
        fea_query = self.memory_banks['franka'](fea_, update=is_train).reshape(b_scn, num_pos_grasp, -1)

        return fea_query, positive_fea, negative_fea


class VGNWrapper(nn.Module):
    base_cfg = {

    }

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.base_cfg, **cfg}
        self.vgn_net = get_network(self.cfg['backbone'])({})

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data (dict): Contains 'geo_info' (scene and gripper grids)
                         and 'grasp_info' (grasp positions, indices, labels, etc.).

        Returns:
            dict: Contains 'vgn_pred' (grasp quality, rotation, width)
                  and memory features for positive/negative examples.
        """
        scene_grid = data['geo_info']['scene']
        vgn_outputs = self.vgn_net(scene_grid)

        if 'full_vol' not in data:
            is_train = data['is_train']
            index = data['grasp_info']['index']
            outputs = {'vgn_pred': self.select(vgn_outputs, index)}
        else:
            outputs = {'vgn_pred': (vgn_outputs['qual'], vgn_outputs['rot'], vgn_outputs['width'])}

        return outputs

    def select(self, out, index):
        """Select specific grasp predictions from the VGN output."""
        qual_out, rot_out, width_out = out['qual'], out['rot'], out[
            'width']  # (2, 1, 80, 80, 80), (2, 4, 80, 80, 80), (2, 1, 80, 80, 80)
        b_scn, num_grasp, _ = index.shape  # index.shape = (2, 24, 3)

        # Flatten index for advanced indexing
        index_flat = index  # shape (B, 24, 3)
        scn_index = torch.arange(b_scn, device=index.device).view(-1, 1)  # shape (B, 1)

        # Select grasp qualities
        label = qual_out[scn_index, :, index[..., 0], index[..., 1], index[..., 2]].squeeze(-1)  # Shape: (B, 24)

        # Select grasp rotations
        rot = rot_out[scn_index, :, index[..., 0], index[..., 1], index[..., 2]]  # Shape: (B, 24, 4)

        # Select grasp widths
        width = width_out[scn_index, :, index[..., 0], index[..., 1], index[..., 2]].squeeze(-1)  # Shape: (B, 24)

        return label, rot, width

name2network = {
    'vgn': VGNWrapper,
    'misc': MISCWrapper,
}