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

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.max_pool_conv(x)
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

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 spatial_kernel_size):
        super().__init__()

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         spatial_kernel_size)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        x = self.gcn(x, A)

        return self.relu(x)

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
    def __init__(self, n_channels, n_classes, **kwargs):
        super(C2F_TCN, self).__init__()

        # load graph
        graph_args = {'layout': 'inhard', 'strategy': 'spatial'}  
        window_size = 120 # 4seconds

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)


        # build networks
        spatial_kernel_size = A.size(0)
        #self.data_bn = nn.BatchNorm1d(n_channels * A.size(1))
        #self.ing = st_gcn(n_channels, 256, spatial_kernel_size)
        n_channels = A.size(0)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(n_channels, 64, spatial_kernel_size),
            st_gcn(64, 128, spatial_kernel_size),
            st_gcn(128, 256, spatial_kernel_size),
            st_gcn(256, 256, spatial_kernel_size)
        ))

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
        #x1 = self.inc(x)
        #print('----------------------------------------')
        #print('input x: ', x.shape)
        # data reshaping
        N, D, T = x.size()
        C = 3
        D = int(D/C)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(N, T, C, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        #print('x reshaped: ', x.shape)
        #----------------------------------------------------------------

        #x1 = self.st_gcn_networks(x, self.A)

        # forwad
        for gcn in self.st_gcn_networks:
            #print('fbefore orward: ', x.shape)
            x = gcn(x, self.A)
            #print('forward: ', x.shape)
        

        # global pooling
        # Apply average pooling along the last two dimensions (spatial dimensions)
        x1 = F.avg_pool2d(x, kernel_size=(1, x.size(-1)))
        x11 = torch.mean(x1, dim=-1)
        #print('global pooling: ', x11.shape)
        
        x2 = self.down1(x11)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        
        '''print('x2: ', x2.shape)
        print('x3: ', x3.shape)
        print('x4: ', x4.shape)
        print('x5: ', x5.shape)
        print('x6: ', x6.shape)'''

        x6 = self.tpp(x6)
        #print('x6 after tpp: ', x6.shape)
        x = self.up(x6, x5)
        #print('x: ', x.shape)
        y2 = self.outcc1(F.relu(x))
        x = self.up1(x, x4)
        y3 = self.outcc2(F.relu(x))
        #print('x: ', x.shape)
        x = self.up2(x, x3)
        y4 = self.outcc3(F.relu(x))
        #print('x: ', x.shape)
        x = self.up3(x, x2)
        y5 = self.outcc4(F.relu(x))
        #print('x: ', x.shape)
        x = self.up4(x, x11)
        y = self.outcc(x)
        '''print('y: ', y.shape)
        print('y5: ', y5.shape)
        print('y4: ', y4.shape)
        print('y3: ', y3.shape)
        print('y2: ', y2.shape)
        print('x: ', x.shape)'''
        return y, [y5, y4, y3, y2], x

