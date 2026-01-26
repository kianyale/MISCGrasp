import os
from builtins import super

import torch.nn.functional as F

from src.miscgrasp.network.MSFUModule import LightFPN3D, MSFUModule
from src.miscgrasp.network.cnn3d import *


def get_network(name):
    models = {
        "baseline_conv": BaselineConv,
        "misc": MISCGrasp,
    }
    return models[name.lower()]


def load_network(path, device):
    """Construct the neural network and load parameters from the specified file.

    Args:
        path: Path to the model parameters. The name must conform to `vgn_name_[_...]`.

    """
    model_name = path.stem.split("_")[1]
    net = get_network(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net


def conv(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def conv_stride(in_channels, out_channels, kernel_size):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2
    )


class Encoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        assert len(filters) == len(kernels), "filters and kernels must have the same length"

        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            in_ch = in_channels if i == 0 else filters[i - 1]
            self.layers.append(conv_stride(in_ch, filters[i], kernels[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, filters, kernels, upsample_sizes):
        super().__init__()
        assert len(filters) == len(kernels) == len(upsample_sizes), \
            "filters, kernels, and upsample_sizes must have the same length"

        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            in_ch = in_channels if i == 0 else filters[i - 1]
            self.layers.append(conv(in_ch, filters[i], kernels[i]))
        self.upsample_sizes = upsample_sizes

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
            x = F.interpolate(x, size=self.upsample_sizes[i])
        return x


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super(UNetDecoder, self).__init__()
        self.upconv = nn.ConvTranspose3d(decoder_channels, decoder_channels // 2, kernel_size=2, stride=2)
        self.conv1 = conv3dBNReLU(encoder_channels + decoder_channels // 2, decoder_channels // 2, kernel_size=3, padding=1)
        self.conv2 = conv3dBNReLU(decoder_channels // 2, decoder_channels // 2, kernel_size=3, padding=1)

    def forward(self, encoder_features, decoder_features):
        decoder_features = self.upconv(decoder_features)
        x = torch.cat([encoder_features, decoder_features], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderMISC(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        assert len(filters) == len(kernels), "filters and kernels must have the same length"

        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            in_ch = in_channels if i == 0 else filters[i - 1]
            self.layers.append(tconv3dBNReLU(in_ch, filters[i], kernels[i], stride=2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MISCGrasp(nn.Module):
    base_cfg = {}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.base_cfg, **cfg}
        # Scene modules
        self.fpn = LightFPN3D(1)
        self.msfu_module = MSFUModule(128)

        # Decoder and output layers
        self.decoder = DecoderMISC(128, [128, 64, 32], [3, 3, 3])
        self.output_layers = self._initialize_output_layers()

    def forward(self, scene):
        b_scn = scene.shape[0]

        # Scene feature extraction
        fea1, fea2, fea3 = self.fpn(scene)  # 'fea1': (B, 32, 40, 40, 40), 'fea2': (B, 64, 20, 20, 20), 'fea3': (B, 128, 10, 10, 10)
        agg_fea = self.msfu_module(fea1, fea2, fea3)  # 'fea1': (B, 128, 40, 40, 40), 'fea2': (B, 128, 20, 20, 20), 'fea3': (B, 128, 10, 10, 10)

        dec_fea = self.decoder(agg_fea)

        outputs = {}
        for key in self.output_layers:
            if key == 'qual':
                outputs[key] = self.output_layers[key](dec_fea)
            elif key == 'rot':
                outputs[key] = F.normalize(self.output_layers[key](dec_fea), dim=1)
            elif key == 'width':
                outputs[key] = self.output_layers[key](dec_fea)
            else:
                raise NotImplementedError('There are only options of `qual`, `width`, and `rot`.')

        outputs['agg_fea1'] = fea1  # NOTE
        outputs['agg_fea2'] = fea2  # NOTE
        outputs['agg_fea3'] = fea3  # NOTE

        return outputs

    def _initialize_align_layers(self):
        return nn.ModuleDict({
            'up_sample1': nn.Conv3d(32, 128, 1, 1, 0),
            'up_sample2': nn.Conv3d(64, 128, 1, 1, 0),
        })

    def _initialize_output_layers(self):
        def conv_block(out_ch):
            return nn.Sequential(
                nn.Conv3d(32, 16, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, out_ch, 5, 1, 2),
            )
        return nn.ModuleDict({
            'qual': conv_block(1),
            'rot': conv_block(4),
            'width': conv_block(1),
        })


class BaselineConv(nn.Module):
    base_cfg = {}
    def __init__(self, cfg):
        super().__init__()
        self.cfg = {**self.base_cfg, **cfg}
        self.scn_encoder = Encoder(1, [16, 32, 64, 128], [5, 3, 3, 3])

        self.decoder = Decoder(128, [128, 64, 32, 16], [3, 3, 3, 5], upsample_sizes=[10, 20, 40, 80])
        self.conv_qual = conv(16, 1, 5)
        self.conv_rot = conv(16, 4, 5)
        self.conv_width = conv(16, 1, 5)

    def forward(self, scn):
        b_scn = scn.shape[0]

        outputs = {}
        scn_fea = self.scn_encoder(scn)
        dec_fea = self.decoder(scn_fea)

        qual_out = self.conv_qual(dec_fea)
        rot_out = F.normalize(self.conv_rot(dec_fea), dim=1)
        width_out = self.conv_width(dec_fea)

        outputs['qual'] = qual_out
        outputs['rot'] = rot_out
        outputs['width'] = width_out

        outputs['agg_fea'] = scn_fea

        return outputs


def visualize(g, filename):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
    ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上
    # ax = plt.axes(projection='3d')
    fig.add_axes(ax)
    ax.voxels(g <= 0, edgecolor='k')
    ax.set_xlabel('X label')  # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    fig.savefig('~/桌面/' + filename + '.png')
    # plt.show()


def rotate_tensor3d(inputs, rotate_beta, rotate_gamma, offset=None, padding_mode='zeros', pre_padding=None):
    """rotate 3D tensor counter-clockwise in z-axis

    Args:
        inputs: torch tensor, [N, C, W, H, D]
        rotate_beta: ndarray,[N]
        rotate_gamma: ndarray,[N]
        offset: None or ndarray, [2, N]
        padding_mode: "zeros" or "border"
        pre_padding: None of float. the valud used for pre-padding such that width == height

    Return:
        outputs: rotated tensor
    """
    device = inputs.device

    if pre_padding is not None:
        lr_pad_w = int((np.max(inputs.shape[2:4]) - inputs.shape[3]) / 2)
        ud_pad_h = int((np.max(inputs.shape[2:4]) - inputs.shape[2]) / 2)
        add_pad = nn.ConstantPad3d((0, 0, lr_pad_w, lr_pad_w, ud_pad_h, ud_pad_h), pre_padding).to(device)
        inputs = add_pad(inputs)

    const_zeros = np.zeros(len(rotate_beta))
    const_ones = np.ones(len(rotate_beta))
    affine1 = np.asarray([[np.cos(rotate_beta), -np.sin(rotate_beta), const_zeros, const_zeros],
                          [np.sin(rotate_beta), np.cos(rotate_beta), const_zeros, const_zeros],
                          [const_zeros, const_zeros, const_ones, const_zeros],
                          [const_zeros, const_zeros, const_zeros, const_ones]])
    affine2 = np.asarray([[const_ones, const_zeros, const_zeros, const_zeros],
                          [const_zeros, np.cos(rotate_gamma), -np.sin(rotate_gamma), const_zeros],
                          [const_zeros, np.sin(rotate_gamma), np.cos(rotate_gamma), const_zeros],
                          [const_zeros, const_zeros, const_zeros, const_ones]])
    affine1 = torch.from_numpy(affine1).permute(2, 0, 1).float().to(device)
    affine2 = torch.from_numpy(affine2).permute(2, 0, 1).float().to(device)
    affine = affine1 @ affine2
    flow_grid = F.affine_grid(affine[:, :3, ...], inputs.size(), align_corners=True).to(device)
    outputs = F.grid_sample(inputs, flow_grid, padding_mode=padding_mode, align_corners=True)
    if offset is not None:
        affine = np.asarray([[const_ones, const_zeros, const_zeros, const_zeros],
                             [const_zeros, const_ones, const_zeros, offset[0]],
                             [const_zeros, const_zeros, const_ones, offset[1]]])
        affine = torch.from_numpy(affine).permute(2, 0, 1).float().to(device)
        flow_grid = F.affine_grid(affine, inputs.size(), align_corners=True).to(device)
        outputs = F.grid_sample(outputs, flow_grid, padding_mode=padding_mode, align_corners=True)
    if pre_padding is not None:
        outputs = outputs[:, :, ud_pad_h:(outputs.shape[2] - ud_pad_h),
                  lr_pad_w:(outputs.shape[3] - lr_pad_w)]
    return outputs


def count_num_trainable_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
