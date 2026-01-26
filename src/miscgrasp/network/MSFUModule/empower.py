import numpy as np
import torch
import torch.nn as nn

from .mos import MixtureOfSoftMax


def group_norm(num_channels):
    return nn.GroupNorm(8, num_channels)

def batch_norm(num_channels):
    return nn.BatchNorm3d(num_channels)

class EmpowerTrans3D(nn.Module):
    def __init__(self, num_heads, num_mixtures, model_dim, key_dim, value_dim,
                 norm_layer=group_norm, kq_transform='conv', v_transform='conv',
                 use_pooling=True, use_concat=False, dropout_rate=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.num_mixtures = num_mixtures
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.use_pooling = use_pooling
        self.use_concat = use_concat

        # Pooling layer
        if use_pooling:
            self.pool_max = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            # self.pool_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
            self.pool_avg = nn.Conv3d(model_dim, model_dim, kernel_size=3, stride=2, padding=1)
            self.up_sample = nn.ConvTranspose3d(num_heads * value_dim, num_heads * value_dim, 3, 2, 1, 1)

        # Key and query transformation
        if kq_transform == 'conv':
            self.q_transform = nn.Conv3d(model_dim, num_heads * key_dim, kernel_size=1)
            nn.init.normal_(self.q_transform.weight, mean=0, std=np.sqrt(2.0 / (model_dim + key_dim)))

        elif kq_transform in ['ffn', 'dffn']:
            dilation = 4 if kq_transform == 'dffn' else 1
            self.q_transform = nn.Sequential(
                nn.Conv3d(model_dim, num_heads * key_dim, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                norm_layer(num_heads * key_dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(num_heads * key_dim, num_heads * key_dim, kernel_size=1),
            )
            nn.init.normal_(self.q_transform[-1].weight, mean=0, std=np.sqrt(1.0 / key_dim))

        else:
            raise NotImplementedError(f"Unsupported kq_transform: {kq_transform}")

        self.k_transform = self.q_transform  # NOTE

        # Value transformation
        if v_transform == 'conv':
            self.v_transform = nn.Conv3d(model_dim, num_heads * value_dim, kernel_size=1)
            nn.init.normal_(self.v_transform.weight, mean=0, std=np.sqrt(2.0 / (model_dim + value_dim)))
        else:
            raise NotImplementedError(f"Unsupported v_transform: {v_transform}")

        # Attention mechanism
        self.attention = MixtureOfSoftMax(n_components=num_mixtures, dim_k=key_dim, mode='dot')

        # Final convolution and normalization
        self.output_conv = nn.Conv3d(num_heads * value_dim, model_dim, kernel_size=1, bias=False)
        self.norm = norm_layer(model_dim)

    def forward(self, x):
        """
        Forward pass for InsightTrans3D.
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
        Return:
            Output tensor after attention and transformation
        """
        residual = x
        batch_size, channels, depth, height, width = x.shape
        num_heads, key_dim, value_dim = self.num_heads, self.key_dim, self.value_dim

        # Apply pooling if enabled
        if self.use_pooling:
            query = self.q_transform(self.pool_avg(x)).view(batch_size * num_heads, key_dim, -1)  # Flatten spatial dimensions
            key = self.k_transform(self.pool_max(x)).view(batch_size * num_heads, key_dim, -1)
            value = self.v_transform(self.pool_max(x)).view(batch_size * num_heads, value_dim, -1)
        else:
            key = self.k_transform(x).view(batch_size * num_heads, key_dim, -1)
            query = self.q_transform(x).view(batch_size * num_heads, key_dim, -1)
            value = self.v_transform(x).view(batch_size * num_heads, value_dim, -1)

        # Compute attention
        output, attention_weights = self.attention(query, key, value)

        if self.use_pooling:
            # Reshape and project output
            output = output.transpose(1, 2).contiguous().view(batch_size, num_heads * value_dim, depth // 2, height // 2, width // 2)
            output = self.up_sample(output)
            output = self.output_conv(output)
        else:
            # Reshape and project output
            output = output.transpose(1, 2).contiguous().view(batch_size, num_heads * value_dim, depth, height, width)
            output = self.output_conv(output)

        # Concatenate or add residual
        if self.use_concat:
            output = torch.cat((self.norm(output), residual), dim=1)
        else:
            output = self.norm(output) + residual

        return output
