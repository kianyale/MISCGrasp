import torch
from torch import nn


class LightFPN3D(nn.Module):
    """
    Feature pyramid network
    """

    def __init__(self, in_ch, out_ch=128):
        super().__init__()

        self.in_planes = 16

        self.out_ch = out_ch

        self.conv1 = conv3dBNReLU(in_ch, 16)
        self.layer1 = self._make_layer(32, stride=2)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        # self.layer4 = self._make_layer(192, stride=2)


    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim)
        layers = (layer1, layer2)

        self.in_planes = dim

        return nn.Sequential(*layers)

    def forward(self, x):
        fea0 = self.conv1(x)
        # print(fea0.shape)
        fea1 = self.layer1(fea0)
        # print(fea1.shape)
        fea2 = self.layer2(fea1)
        # print(fea2.shape)
        fea3 = self.layer3(fea2)
        # print(fea3.shape)
        # fea4 = self.layer4(fea3)

        return fea1, fea2, fea3
        # return fea2, fea3, fea4


class conv3dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
        self.BN = torch.nn.BatchNorm3d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        bn = self.BN(self.conv(x))
        return self.relu(bn)


class conv3dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3dBNReLU(in_planes, planes, 3, stride=stride, padding=1)
        self.conv2 = conv3dBN(planes, planes, 3, stride=1, pad=1)

        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = conv3dBN(in_planes, planes, 3, stride=stride, pad=1)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


if __name__ == '__main__':
    from torchinfo import summary

    fpn = LightFPN3D(in_ch=2, out_ch=128)
    summary(fpn, (8, 2, 60, 60, 60))


