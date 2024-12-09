# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
from graph import Graph
import numpy as np
from model.modules import EncodingBlock

class InfoGCN(nn.Module):

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
                 num_heads,
                 hidden_dim,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.num_heads = num_heads 
        num_point= 21
        self.num_point = num_point
       
        A = np.stack([np.eye(num_point)] * self.num_heads, axis=0)

        self.to_joint_embedding = nn.Linear(kernel_size, out_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, out_channels))
        self.data_bn = nn.BatchNorm1d(out_channels * num_point)
        self.l1 = EncodingBlock(out_channels, out_channels, A, hidden_dim)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)


    def forward(self, x, A):
        #print('inside infoGcn: --------------------------------------------')

        #print("self.kernel_size: ", self.kernel_size, "A: ", A.shape)
        assert A.size(0) == self.kernel_size
        #print('x:', x.shape)

        x = self.conv(x)
        #print('x after conv: ', x.shape)
        
        n, kc, t, v = x.size()
        
        x = x.view(n*t, self.kernel_size, kc//self.kernel_size, v)
        #print('x:', x.shape)
        x = torch.einsum('nkcv,kvw->nwk', (x, A))  #[36800, 21, 3]
        #print('x = A*X:', x.shape) 
        
        x = self.to_joint_embedding(x)   #[36800, 21, 64]
        #print('Em(x):', x.shape)

        x += self.pos_embedding[:, :self.num_point]
        #print('x = Em(x) + PE:', x.shape)  #[36800, 21, 64]
        n_2, w, k = x.size()

        x = x.view(n, v*k, t)             #[5, 21*64, 7360]
        #print('x reshape:', x.shape)

        x = self.data_bn(x)
        #print('x = bn(x):', x.shape)

        x = x.view(n, k, t, v)
        #print('x reshape:', x.shape)

        # encoding block starts
        x = self.l1(x)

        return x.contiguous()
