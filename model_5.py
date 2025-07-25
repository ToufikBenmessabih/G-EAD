import torch.nn.functional as F
import math
import torch.nn as nn
import torch
from functools import partial
import torchvision.models as mdels

nonlinearity = partial(F.relu, inplace=True)

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        #print('inin: ', x.shape)
        x = self.conv(x)
        #print('outout: ', x.shape)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        #print('in: ', x.shape)
        x = self.max_pool_conv(x)
        #print('out: ', x.shape)
        return x

class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diff // 2, diff - diff //2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TPPblock(nn.Module):
    def __init__(self, in_channels):
        super(TPPblock, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )

    def forward(self, x):
        self.in_channels, t = x.size(1), x.size(2)
        self.layer1 = F.upsample(
            self.conv(self.pool1(x)), size=t, mode="linear", align_corners=True
        )
        self.layer2 = F.upsample(
            self.conv(self.pool2(x)), size=t, mode="linear", align_corners=True
        )
        self.layer3 = F.upsample(
            self.conv(self.pool3(x)), size=t, mode="linear", align_corners=True
        )
        self.layer4 = F.upsample(
            self.conv(self.pool4(x)), size=t, mode="linear", align_corners=True
        )

        out = torch.cat([self.layer1, self.layer2,
                         self.layer3, self.layer4, x], 1)

        return out


class C2F_TCN(nn.Module):
    '''
        Features are extracted at the last layer of decoder. 
    '''
    def __init__(self, n_channels, n_classes):
        super(C2F_TCN, self).__init__()
        self.inc = inconv(n_channels, 256)
        self.down1 = down(256, 256)
        self.down2 = down(256, 256)
        self.down3 = down(256, 128)
        self.down4 = down(128, 128)
        self.down5 = down(128, 128)
        self.down6 = down(128, 128)
        self.up = up(260, 128)
        self.outcc0 = outconv(128, n_classes)
        self.up0 = up(256, 128)
        self.outcc1 = outconv(128, n_classes)
        self.up1 = up(256, 128)
        self.outcc2 = outconv(128, n_classes)
        self.up2 = up(384, 128)
        self.outcc3 = outconv(128, n_classes)
        self.up3 = up(384, 128)
        self.outcc4 = outconv(128, n_classes)
        self.up4 = up(384, 128)
        self.outcc = outconv(128, n_classes)
        self.tpp = TPPblock(128)
        self.weights = torch.nn.Parameter(torch.ones(5)) #6

    def forward(self, x):

        '''# Check for NaNs in raw data
        if torch.isnan(x).any():
            print("NaNs found in raw input data")
            print("Indices of NaNs:", torch.nonzero(torch.isnan(x)))
        # Replace NaNs with zeros
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        # Validate after replacement
        if torch.isnan(x).any():
            print("NaNs found after replacement")
            # You may want to take additional actions here'''
    
        #print('input: ', x.shape)
        x1 = self.inc(x)
        #print('x1: ', x1.shape)
        x2 = self.down1(x1)
        #print('x2: ', x2.shape)
        x3 = self.down2(x2)
        #print('x3: ', x3.shape)
        x4 = self.down3(x3)
        #print('x4: ', x4.shape)
        x5 = self.down4(x4)
        #print('x5: ', x5.shape)
        x6 = self.down5(x5)
        #print('x6: ', x6.shape)

        x6 = self.tpp(x6)

        x = self.up(x6, x5)
        y2 = self.outcc1(F.relu(x))
        x = self.up1(x, x4)
        y3 = self.outcc2(F.relu(x))
        x = self.up2(x, x3)
        y4 = self.outcc3(F.relu(x))
        x = self.up3(x, x2)
        y5 = self.outcc4(F.relu(x))
        x = self.up4(x, x1)
        y = self.outcc(x)
        return y, [y5, y4, y3, y2], x

