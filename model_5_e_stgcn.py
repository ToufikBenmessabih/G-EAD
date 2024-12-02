import torch.nn.functional as F
import math
import torch.nn as nn
import torch
from functools import partial
import torchvision.models as mdels
from torch.nn.parameter import Parameter

from tgcn import ConvTemporalGraphical
from graph import Graph

nonlinearity = partial(F.relu, inplace=True)
import numpy as np
from IPython.core.debugger import set_trace

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
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
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

        '''print('layer1: ', self.layer1.shape)
        print('layer2: ', self.layer2.shape)
        print('layer3: ', self.layer3.shape)
        print('layer4: ', self.layer4.shape)'''

        out = torch.cat([self.layer1, self.layer2,
                         self.layer3, self.layer4, x], 1)

        return out

class down(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        
        self.max_pool = nn.MaxPool1d(2)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


 
    def forward(self, x, A, MaxPool):

        x = self.gcn(x, A)

        if MaxPool:
            # Permute to bring the third dimension to the second position
            x_permuted = x.permute(0, 1, 3, 2)  # Shape: [10, 256, 21, 7360]

            # Reshape to merge the last two dimensions
            N, C, M, L = x_permuted.shape
            x_reshaped = x_permuted.view(N * C, M, L)  # Shape: [2560, 21, 7360]

            x = self.max_pool(x_reshaped)

            # Reshape back to the original dimensions
            _, M_pooled, L_pooled = x.shape
            x_pooled_reshaped = x.view(N, C, M_pooled, L_pooled)  # Shape: [10, 256, 21, 3680]

            # Permute back to the original shape
            x = x_pooled_reshaped.permute(0, 1, 3, 2)  # Shape: [10, 256, 3680, 21]

        
        
        x = self.conv_block(x)

        x_pool = F.avg_pool2d(x, kernel_size=(1, x.size(-1)))
        x_pool = torch.mean(x, dim=-1)

        return x, x_pool


class C2F_TCN(nn.Module):
    r"""Spatial temporal graph convolutional networks. just for the first layers

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, T, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence = nbr frames,
            :math:`V_{in}` is the number of graph nodes = 21,
            :math:`M_{in}` is the number of instance (persons) in a frame = 1 person.
    """
    def __init__(self, n_channels, n_classes):
        super(C2F_TCN, self).__init__()

        # load graph
        graph_args = {'layout': 'inhard_72', 'strategy': 'spatial'}  
        window_size = 120 # 4seconds

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)


        # build networks
        spatial_kernel_size = A.size(0)
        print('spatial_kernel_size: ', spatial_kernel_size)
        temporal_kernel_size = 31
        print('temporal_kernel_size: ', temporal_kernel_size)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        n_channels = A.size(0)

        self.ing = down(n_channels, 256, kernel_size)

        self.down1 = down(256, 256, kernel_size)
        self.down2 = down(256, 256, kernel_size)
        self.down3 = down(256, 128, kernel_size)
        self.down4 = down(128, 128, kernel_size)
        self.down5 = down(128, 128, kernel_size)
        self.down6 = down(128, 128, kernel_size)
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
        # data reshaping
        N, D, T = x.size()
        C = 3
        D = int(D/C)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(N, T, C, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        #----------------------------------------------------------------

        x1, x1_pool = self.ing(x, self.A, False)
        
        
        x2, x2_pool = self.down1(x1, self.A, True)
        #print('x2: ', x2.shape)
        x3, x3_pool = self.down2(x2, self.A, True)
        #print('x3: ', x3.shape)
        x4, x4_pool = self.down3(x3, self.A, True)
        #print('x4: ', x4.shape)
        x5, x5_pool = self.down4(x4, self.A, True)
        #print('x5: ', x5.shape)
        x6, x6_pool = self.down5(x5, self.A, True)
        #print('x6: ', x6.shape)

        

        x6 = self.tpp(x6_pool)
        
        x = self.up(x6, x5_pool)
        y2 = self.outcc1(x)
        x = self.up1(x, x4_pool)
        y3 = self.outcc2(x)
        x = self.up2(x, x3_pool)
        y4 = self.outcc3(x)
        x = self.up3(x, x2_pool)
        y5 = self.outcc4(x)
        x = self.up4(x, x1_pool)
        y = self.outcc(x)
        return y, [y5, y4, y3, y2], x

