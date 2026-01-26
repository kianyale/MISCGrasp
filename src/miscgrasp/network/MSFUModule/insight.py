import torch
from torch import nn


class InsightTrans3D(nn.Module):
    def __init__(self, high_channels, low_channels, downsample_factor, use_upsample=True):
        """
        3D feature transformation and insight module.

        Args:
            high_channels (int): Number of channels in the high-level feature map.
            low_channels (int): Number of channels in the low-level feature map.
            use_upsample (bool): Whether to upsample the low-level feature map.
        """
        super().__init__()
        self.use_upsample = use_upsample

        # High-level processing layers
        self.high_conv = nn.Conv3d(high_channels, high_channels, kernel_size=3, padding=1, bias=False)
        self.high_norm = nn.GroupNorm(8, high_channels)

        # Low-level global pooling and transformation
        self.low_conv1x1 = nn.Conv3d(low_channels, high_channels, kernel_size=1, bias=False)
        self.low_norm = nn.GroupNorm(8, high_channels)

        # Upsample or downsample low-level features
        if use_upsample:
            self.low_upsample = nn.ConvTranspose3d(low_channels, high_channels, kernel_size=4, stride=2, padding=1,
                                                   bias=False)
            self.low_upsample_norm = nn.GroupNorm(8, high_channels)
        else:
            self.low_downsample = nn.Sequential(*[nn.Conv3d(low_channels if i == 0 else high_channels,
                                                            high_channels,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            bias=False) for i in range(downsample_factor)])
            self.low_downsample_norm = nn.GroupNorm(8, high_channels)
        self.low_transformed_norm = nn.GroupNorm(8, high_channels)


        # Refinement and combination layers
        self.relu = nn.ReLU(inplace=True)
        self.combine_conv = nn.Conv3d(high_channels, high_channels, kernel_size=3, padding=1, bias=False)

        self.output_conv = nn.Conv3d(high_channels, high_channels, kernel_size=3, padding=1, bias=False)


    def forward(self, x_high, x_low):
        """
        Forward pass for RenderTrans3D.

        Args:
            x_high (torch.Tensor): High-level feature map of shape (B, C_high, D_high, H_high, W_high).
            x_low (torch.Tensor): Low-level feature map of shape (B, C_low, D_low, H_low, W_low).

        Returns:
            torch.Tensor: Refined feature map of shape (B, C_high, D_high, H_high, W_high).
        """
        batch_size, low_channels, depth, height, width = x_low.shape

        # Global pooling and feature transformation for low-level features
        low_global = nn.functional.adaptive_avg_pool3d(x_low, output_size=(1, 1, 1))  # Global pooling
        low_global = self.low_conv1x1(low_global)  # Transform to high channel dimensions
        low_global = self.low_norm(low_global)
        low_global = self.relu(low_global)

        # Process high-level features
        high_processed = self.high_conv(x_high)
        high_processed = self.high_norm(high_processed)

        # Attention-based refinement
        attention_map = self.combine_conv(high_processed * low_global)

        # Upsample or downsample low-level features and combine
        if self.use_upsample:
            low_transformed = self.low_upsample(x_low)
        else:
            low_transformed = self.low_downsample(x_low)

        # Combine transformed low-level and high-level features
        output = self.relu(self.low_transformed_norm(low_transformed + attention_map))
        output = self.output_conv(output)

        return output
