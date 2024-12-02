import torch.nn.functional as F
import math
import torch.nn as nn
import torch
from functools import partial
import torchvision.models as mdels
from torch.nn.parameter import Parameter

from tagcn import AdaptiveConvTemporalGraphical
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
    def __init__(self, in_ch, out_ch, kernel_size):
        super(outconv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)


    def forward(self, x, A):
        x = self.conv(x)

        #avg pool
        x_pool = F.avg_pool2d(x, kernel_size=(1, x.size(-1)))
        x_pool = torch.mean(x, dim=-1)
        
        return x_pool

class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        '''self.gcn = ConvTemporalGraphical(in_channels, in_channels,
                                         kernel_size[1])'''
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x1, x2, A):
        #print('x1: ', x1.shape)
        #print('x2: ', x2.shape)

        # Permute to bring the third dimension to the second position
        x_permuted = x1.permute(0, 1, 3, 2)  # Shape: [10, 132, 21, 230]
        #print('x_permuted: ', x_permuted.shape)

        # Reshape to merge the last two dimensions
        N, C, M, L = x_permuted.shape
        x_reshaped = x_permuted.view(N * C, M, L)  # Shape: [1320, 21, 230]
        #print('x_reshaped: ', x_reshaped.shape)

        x1 = self.up(x_reshaped)
        #print('x1 (up): ', x1.shape)

        # input is CHW
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])
        x1 = F.pad(x1, [diff // 2, diff - diff //2])
        #print('x1 (pad): ', x1.shape)

        # Reshape back to the original dimensions
        _, M_pooled, L_pooled = x1.shape
        x_pooled1_reshaped = x1.view(N, C, M_pooled, L_pooled)  # Shape: [10, 132, 21, 460]
        
        # Permute back to the original shape
        x1 = x_pooled1_reshaped.permute(0, 1, 3, 2)  # Shape: [10, 132, 460, 21]
        
        x = torch.cat([x2, x1], dim=1)
        #print('x (cat): ', x.shape)
        #gcn
        #x = self.gcn(x, A)
        x = self.conv(x)
        #print('x (out): ', x.shape)
        return x

class TPPblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TPPblock, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )

    def forward(self, x):

        #print('input(x) ttp: ', x.shape)
        self.in_channels, t = x.size(1), x.size(2)
        #print('in_channels: ', self.in_channels,'t: ', t)

        # Permute to bring the third dimension to the second position
        x_permuted = x.permute(0, 1, 3, 2)  # Shape: [10, 128, 21, 230]
        #print('x_permuted: ', x_permuted.shape)

        # Reshape to merge the last two dimensions
        N, C, M, L = x_permuted.shape
        x_reshaped = x_permuted.view(N * C, M, L)  # Shape: [1280, 21, 230]
        #print('x_reshaped: ', x_reshaped.shape)

        x_pool1 = self.pool1(x_reshaped)
        x_pool2 = self.pool2(x_reshaped)
        x_pool3 = self.pool3(x_reshaped)
        x_pool4 = self.pool4(x_reshaped)
        '''print('x_pool1: ', x_pool1.shape)
        print('x_pool2: ', x_pool2.shape)
        print('x_pool3: ', x_pool3.shape)
        print('x_pool4: ', x_pool4.shape)'''


        # Reshape back to the original dimensions
        _, M_pooled, L_pooled = x_pool1.shape
        x_pooled1_reshaped = x_pool1.view(N, C, M_pooled, L_pooled)  # Shape: [10, 128, 21, 115]
        
        _, M_pooled, L_pooled = x_pool2.shape
        x_pooled2_reshaped = x_pool2.view(N, C, M_pooled, L_pooled)
        
        _, M_pooled, L_pooled = x_pool3.shape
        x_pooled3_reshaped = x_pool3.view(N, C, M_pooled, L_pooled)
        
        _, M_pooled, L_pooled = x_pool4.shape
        x_pooled4_reshaped = x_pool4.view(N, C, M_pooled, L_pooled)


        # Permute back to the original shape
        x_pool1 = x_pooled1_reshaped.permute(0, 1, 3, 2)  # Shape: [10, 128, 115, 21]
        x_pool2 = x_pooled2_reshaped.permute(0, 1, 3, 2)
        x_pool3 = x_pooled3_reshaped.permute(0, 1, 3, 2)
        x_pool4 = x_pooled4_reshaped.permute(0, 1, 3, 2)
        #print('x_pool1 after reshape: ', x_pool1.shape)

        #tcn
        x1_conv = self.conv(x_pool1)
        x2_conv = self.conv(x_pool1)
        x3_conv = self.conv(x_pool1)
        x4_conv = self.conv(x_pool1)

        # Remove the unnecessary dimension (1)
        x1_conv = x1_conv.squeeze(1)  # Now x1_conv has size [10, 115, 21]
        x2_conv = x2_conv.squeeze(1)
        x3_conv = x3_conv.squeeze(1)
        x4_conv = x4_conv.squeeze(1)

        # permute and apply upsampling on last dim
        x1_conv = x1_conv.permute(0, 2, 1)  # Shape: [10, 21, 115]
        x2_conv = x2_conv.permute(0, 2, 1)
        x3_conv = x3_conv.permute(0, 2, 1)
        x4_conv = x4_conv.permute(0, 2, 1)

        self.layer1 = F.upsample(
            x1_conv, size=t, mode="linear", align_corners=True
        )
        self.layer2 = F.upsample(
            x2_conv, size=t, mode="linear", align_corners=True
        )
        self.layer3 = F.upsample(
            x3_conv, size=t, mode="linear", align_corners=True
        )
        self.layer4 = F.upsample(
            x4_conv, size=t, mode="linear", align_corners=True
        )

        # permute back
        x_up1 = self.layer1.permute(0, 2, 1)  # Shape: [10, 230, 21]
        x_up2 = self.layer2.permute(0, 2, 1)
        x_up3 = self.layer3.permute(0, 2, 1)
        x_up4 = self.layer4.permute(0, 2, 1)
        #print('x1 upsampled: ', x_up1.shape)

        # add a dim
        x_up1 = x_up1.unsqueeze(1)
        x_up2 = x_up2.unsqueeze(1)
        x_up3 = x_up3.unsqueeze(1)
        x_up4 = x_up4.unsqueeze(1)
        #print('x1 final: ', x_up1.shape)
        
        out = torch.cat([x_up1, x_up2, x_up3, x_up4, x], 1)
        #print('out ttp: ', out.shape)

        return out

class down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        self.gcn = AdaptiveConvTemporalGraphical(in_channels, out_channels,
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

        return x


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
        graph_args = {'layout': 'inhard', 'strategy': 'spatial'}  
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
        self.up = up(260, 128, kernel_size)
        self.outcc1 = outconv(128, n_classes, kernel_size)
        self.up1 = up(256, 128, kernel_size)
        self.outcc2 = outconv(128, n_classes, kernel_size)
        self.up2 = up(384, 128, kernel_size)
        self.outcc3 = outconv(128, n_classes, kernel_size)
        self.up3 = up(384, 128, kernel_size)
        self.outcc4 = outconv(128, n_classes, kernel_size)
        self.up4 = up(384, 128, kernel_size)
        self.outcc = outconv(128, n_classes, kernel_size)
        self.tpp = TPPblock(128, 1, kernel_size)
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

        x1 = self.ing(x, self.A, False)
        
        
        x2 = self.down1(x1, self.A, True)
        #print('x2: ', x2.shape)
        x3 = self.down2(x2, self.A, True)
        #print('x3: ', x3.shape)
        x4 = self.down3(x3, self.A, True)
        #print('x4: ', x4.shape)
        x5 = self.down4(x4, self.A, True)
        #print('x5: ', x5.shape)
        x6 = self.down5(x5, self.A, True)
        #print('x6: ', x6.shape)

        # input [10, 128, 230, 21]
        x6 = self.tpp(x6)
        #print('output TTP: ', x6.shape) # [10, 132, 230, 21]


        x = self.up(x6, x5, self.A)
        y2 = self.outcc1(x, self.A)
        x = self.up1(x, x4, self.A)
        y3 = self.outcc2(x, self.A)
        x = self.up2(x, x3, self.A)
        y4 = self.outcc3(x, self.A)
        x = self.up3(x, x2, self.A)
        y5 = self.outcc4(x, self.A)
        x = self.up4(x, x1, self.A)
        y = self.outcc(x, self.A)

        
        #print('output x: ', x.shape) 
        #print('output y: ', y.shape) 
        return y, [y5, y4, y3, y2], x

