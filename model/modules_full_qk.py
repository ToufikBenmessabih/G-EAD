import math

import torch
import numpy as np

from torch import nn, einsum
from torch.autograd import Variable
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from einops import rearrange, repeat

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_heads):
        super(SelfAttention, self).__init__()
        '''
        num_head:  2
        in_channels:  132
        hidden:  44
        inner_dim:  88
        '''
        self.scale = hidden_dim ** -0.5 # A/racine(hidden_dim)
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim*2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x):
        y = rearrange(x, 'n c t v -> n t v c').contiguous() # (n, t, v, dim) [5, 230, 21, 132]
        y = self.ln(y)             # [5, 230, 21, 132]
        y = self.to_qk(y)          # (n, t, v, 176) [5, 230, 21, 176]   
        qk = y.chunk(2, dim=-1)    # q and k, each with shape (n, t, v, 88)
        q, k = qk                                           # Each: [b, t, v, d]

        # Reshape (combine batch and time dimensions) and duplicate for 2 heads
        q = q.reshape(-1, q.shape[2], q.shape[3])  # Shape: [b*t, v, d]
        k = k.reshape(-1, k.shape[2], k.shape[3])  # Shape: [b*t, v, d]
        q = q.unsqueeze(1).repeat(1, 2, 1, 1)  # Duplicate for heads: [b*t, 2, v, d]
        k = k.unsqueeze(1).repeat(1, 2, 1, 1)  # Same for K: [b*t, 2, v, d]


        #print('Q:', q.shape)       # [1150, 2, 21, 88]
        #print('K:', k.shape)       # [1150, 2, 21, 88]
        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k)*self.scale  #(b, 2, i, j)
        attn = dots.softmax(dim=-1).float()  # [1150, 2, 21, 21] 
        
        return attn
    
class SA_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SA_GC, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head= A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

        rel_channels = in_channels // 3
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)


    def forward(self, x, attn=None):
        N, C, T, V = x.size()  # [5, 132, T = 230, 21]

        #print("num_head: ", self.num_head) # 2

        out = None
        if attn is None:
            attn = self.attn(x)  # [1150, 2, 21, 21] 

            # Move tensor to CPU before converting to NumPy
            # attn_npy = attn.cpu().numpy()
            # Save as .npy file
            # np.save('attention_matrix.npy', attn_npy)

        A = attn * self.shared_topology.unsqueeze(0)  # [b*t, h, v, v] [1150, 2, 21, 21]

 
        for h in range(self.num_head):
            A_h = A[:, h, :, :] # (nt)vv----Selecting Attention Matrix for the Current Head
            feature = rearrange(x, 'n c t v -> (n t) v c')
            z = A_h@feature  # applies the attention weights from A_h to the feature representations in x.
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)  # [5, 132, 230, 21]
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)  # [5, 132, 230, 21]

        return out

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(EncodingBlock, self).__init__()
        self.agcn = SA_GC(in_channels, out_channels, A)

        self.relu = nn.ReLU(inplace=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

    def forward(self, x, attn=None):
        #print('input encoding block: ', x.shape)
        y = self.relu(self.agcn(x, attn) + self.residual(x))
        #print('output encoding block: ', y.shape)
        return y

