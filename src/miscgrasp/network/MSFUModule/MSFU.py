import torch
import torch.nn as nn

from .empower import EmpowerTrans3D
from .insight import InsightTrans3D


class MSFUModule(nn.Module):
    def __init__(self, feature_dim, with_norm='group_norm', upsample_method='trilinear'):
        """
        Feature Pyramid Transformer for 3D data with optional normalization and upsampling methods.

        Args:
            feature_dim (int): Number of output channels for feature maps.
            with_norm (str): Normalization method ('batch_norm', 'group_norm', 'layer_norm', 'instance_norm', 'none').
            upsample_method (str): Upsampling method ('trilinear', 'nearest', 'bilinear').
        """
        super().__init__()
        self.feature_dim = feature_dim
        assert upsample_method in ['trilinear', 'nearest', 'bilinear'], "Unsupported upsample method."
        assert with_norm in ['instance_norm', 'layer_norm', 'group_norm', 'batch_norm', 'none'], "Unsupported norm method."

        # Define normalization layer
        self.norm = self._select_norm_layer(with_norm)

        # Self-Transformation layers
        self.et_p4 = EmpowerTrans3D(1, 2, feature_dim, feature_dim, feature_dim, use_pooling=False)

        # Render-Transformation layers
        self.it_p4_p3 = InsightTrans3D(feature_dim, feature_dim, downsample_factor=1, use_upsample=False)
        self.it_p4_p2 = InsightTrans3D(feature_dim, feature_dim, downsample_factor=2, use_upsample=False)

        # 1x1 Convolution for feature alignment
        self.fpn_p4_1x1 = self._create_conv_block(128, feature_dim, kernel_size=1)
        self.fpn_p3_1x1 = self._create_conv_block(64, feature_dim, kernel_size=1)
        self.fpn_p2_1x1 = self._create_conv_block(32, feature_dim, kernel_size=1)

        # Final feature transformation layers
        self.out_p4 = self._create_conv_block(feature_dim * 4, feature_dim, kernel_size=3)

        # Initialize weights
        self._initialize_weights()

    def _select_norm_layer(self, norm_type):
        """Select the appropriate normalization layer based on the type."""
        if norm_type == 'instance_norm':
            return nn.InstanceNorm3d
        elif norm_type == 'layer_norm':
            return nn.LayerNorm
        elif norm_type == 'batch_norm':
            return nn.BatchNorm3d
        elif norm_type == 'group_norm':
            return lambda num_channels: nn.GroupNorm(8, num_channels)
        else:
            return None

    def _create_conv_block(self, in_channels, out_channels, kernel_size):
        """Create a convolutional block with optional normalization."""
        if kernel_size != 1:
            layers = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)]
        else:
            layers = [nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)]
        if self.norm:
            layers.append(self.norm(out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights for all convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, res2, res3, res4):
        """
        Forward pass for MSFUModule.

        Args:
            res2 (torch.Tensor): Low-level feature map (shape: B, 32, D, H, W).
            res3 (torch.Tensor): Mid-level feature map (shape: B, 64, D/2, H/2, W/2).
            res4 (torch.Tensor): High-level feature map (shape: B, 128, D/4, H/4, W/4).

        Returns:
            tuple: Transformed feature maps (p2, p3, p4).
        """
        # Apply 1x1 convolutions for feature alignment
        fpn_p4 = self.fpn_p4_1x1(res4)
        fpn_p3 = self.fpn_p3_1x1(res3)
        fpn_p2 = self.fpn_p2_1x1(res2)

        # Multi-level feature transformation
        p4_out = torch.cat((
            self.et_p4(fpn_p4),
            self.it_p4_p3(fpn_p4, fpn_p3),
            self.it_p4_p2(fpn_p4, fpn_p2),
            fpn_p4
        ), dim=1)

        # Final feature transformation
        res = self.out_p4(p4_out)

        return res


if __name__ == '__main__':
    from torchinfo import summary

    model = MSFUModule(feature_dim=128, with_norm='group_norm', upsample_method='trilinear')
    summary(model, input_size=[(1, 32, 40, 40, 40), (1, 64, 20, 20, 20), (1, 128, 10, 10, 10)])

