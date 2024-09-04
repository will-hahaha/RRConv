import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RRConv import RectConv2d

class RectB(nn.Module):
    def __init__(self, in_planes, flag=False):
        super(RectB, self).__init__()
        self.flag = flag
        self.conv1 = RectConv2d(in_planes, in_planes, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = RectConv2d(in_planes, in_planes, 3, 1, 1)

    def forward(self, x, epoch, label1, label2, nx1, ny1, nx2, ny2):
        res = self.conv1(x, epoch, label1, nx1, ny1)
        res = self.relu(res)
        res = self.conv2(res, epoch, label2, nx2, ny2)
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
        self.rb1 = RectB(32)
        self.down1 = ConvDown(32)
        self.rb2 = RectB(64)
        self.down2 = ConvDown(64)
        self.rb3 = RectB(128)
        self.up1 = ConvUp(128)
        self.rb4 = RectB(64)
        self.up2 = ConvUp(64)
        self.rb5 = RectB(32)
        self.tail_conv = nn.Conv2d(32, 8, 3, 1, 1)

    def forward(self, pan, lms, epoch, nx1 = 3, nx2 = 3, nx3 = 3, nx4 = 3, nx5 = 3, nx6 = 3, nx7 = 3, nx8 = 3, nx9 = 3, nx10 = 3, ny1 = 3, ny2 = 3, ny3 = 3, ny4 = 3, ny5 = 3, ny6 = 3, ny7 = 3, ny8 = 3, ny9 = 3, ny10 = 3):
        x1 = torch.cat([pan, lms], dim=1)
        x1 = self.head_conv(x1)
        x1 = self.rb1(x1, epoch, 1, 2, nx1, ny1, nx2, ny2)
        x2 = self.down1(x1)
        x2 = self.rb2(x2, epoch, 3, 4, nx3, ny3, nx4, ny4)
        x3 = self.down2(x2)
        x3 = self.rb3(x3, epoch, 5, 6, nx5, ny5, nx6, ny6)
        x4 = self.up1(x3, x2)
        del x2
        x4 = self.rb4(x4, epoch, 7, 8, nx7, ny7, nx8, ny8)
        x5 = self.up2(x4, x1)
        del x1
        x5 = self.rb5(x5, epoch, 9, 10, nx9, ny9, nx10, ny10)
        x5 = self.tail_conv(x5)
        return lms + x5
