import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from AECRNet.AECR.models.DCNv2.dcn_v2 import DCN

class DCNB(nn.Module):
    def __init__(self, in_planes, flag=False):
        super(DCNB, self).__init__()
        self.flag = flag
        self.block = nn.Sequential(
            DCN(in_planes, in_planes, 3, 1, 1),
            nn.ReLU(inplace=True),
            DCN(in_planes, in_planes, 3, 1, 1)
        )
    def forward(self, x):
        res = self.block(x)
        x = x + res
        return x
class ConvDown(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if dsconv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1)
            )

    def forward(self, x):
        return self.conv(x)


class ConvUp(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2, 0)
        if dsconv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2, bias=False),
                nn.Conv2d(in_channels // 2, in_channels // 2, 1, 1, 0)
            )
        else:
            self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1)

    def forward(self, x, y):
        x = F.leaky_relu(self.conv1(x))
        x = x + y
        x = F.leaky_relu(self.conv2(x))
        return x


class RECTNET(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(RECTNET, self).__init__()
        self.head_conv = nn.Conv2d(9, 32, 3, 1, 1)
        self.rb1 = DCNB(32)
        self.down1 = ConvDown(32)
        self.rb2 = DCNB(64)
        self.down2 = ConvDown(64)
        self.rb3 = DCNB(128)
        self.up1 = ConvUp(128)
        self.rb4 = DCNB(64)
        self.up2 = ConvUp(64)
        self.rb5 = DCNB(32)
        self.tail_conv = nn.Conv2d(32, 8, 3, 1, 1)

    def forward(self, pan, lms):
        x1 = torch.cat([pan, lms], dim=1)
        x1 = self.head_conv(x1)
        x1 = self.rb1(x1)
        x2 = self.down1(x1)
        x2 = self.rb2(x2)
        x3 = self.down2(x2)
        x3 = self.rb3(x3)
        x4 = self.up1(x3, x2)
        del x2
        x4 = self.rb4(x4)
        x5 = self.up2(x4, x1)
        del x1
        x5 = self.rb5(x5)
        x5 = self.tail_conv(x5)
        return lms + x5
