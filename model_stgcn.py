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


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

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
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        #print("kernel_size[1]: ", kernel_size[1])

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        

        if not residual:
            print('no residual')
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            print('same channels')
            self.residual = lambda x: x

        else:
            print('res is conv2D + batchnorm2D')
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x, A):

        #print('input x: ', x.shape)
        res= self.residual(x)
        x = self.gcn(x, A)
        #print('gcn x: ', x.shape)
        x = self.tcn(x) + res
        #print('tcn x: ', x.shape)

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
        #graph_args = {'layout': 'inhard', 'strategy': 'spatial'}  
        #graph_args = {'layout': 'inhard_72', 'strategy': 'spatial'} 
        graph_args = {'layout': 'IKEA', 'strategy': 'spatial'} 
        window_size = 120 # 4seconds not used

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)


        # build networks
        spatial_kernel_size = A.size(0)
        print('spatial_kernel_size: ', spatial_kernel_size)
        temporal_kernel_size = 31
        print('temporal_kernel_size: ', temporal_kernel_size)
        spatial_kernel_size = 1
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        n_channels = 3 # 3D sk

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(n_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 128, kernel_size, 1, residual=False),
            st_gcn(128, 256, kernel_size, 1, residual=False),
            st_gcn(256, 256, kernel_size, 1, residual=False)
        ))

        # fcn for prediction
        self.fcn = outconv(256, n_classes)

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

        # forwad
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)
            #print('x forwad: ', x.shape)
        

        # global pooling
        # Apply average pooling along the last two dimensions (spatial dimensions)
        #print('x gcn output: ', x1.shape)
        x1 = F.avg_pool2d(x, kernel_size=(1, x.size(-1)))
        x11 = torch.mean(x1, dim=-1)
        #print('global pooling: ', x11.shape)

        y = self.fcn(x11)
        return  y

