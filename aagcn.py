# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
from graph import Graph
import numpy as np
from model.unit_aagcn import unit_aagcn

class AAGCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True):
        super(AAGCN_unit, self).__init__()
        self.gcn1 = unit_aagcn(in_channels, out_channels, A, adaptive=adaptive, attention=attention)
        self.relu = nn.ReLU(inplace=True)

        self.attention = attention

    def forward(self, x):
        y = self.relu(self.gcn1(x))
        return y
    
class AAGCN(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True, adaptive=True, attention=True):
        super().__init__()

        self.kernel_size = kernel_size
        num_head = 3
        num_point=21
        self.num_point = num_point

        # load graph
        graph_args = {'layout': 'inhard', 'strategy': 'spatial'}  

        self.graph = Graph(**graph_args)
        A = self.graph.A
       
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)

        self.l1 = AAGCN_unit(in_channels,out_channels, A, residual=False, adaptive=adaptive, attention=attention)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)


    def forward(self, x, A):
        #print('inside agcn: --------------------------------------------')

        #print('x (input):', x.shape)
        N, C, T, V = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(N, V * C, T)
        #print('x (befor bn):', x.shape)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        #x = self.conv(x)
        #print('x (reshaped): ', x.shape)
        

        x = self.l1(x)

        return x.contiguous()
