from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    def __init__(self, leaky_rate, node_cnt, dropout_rate):
        super().__init__()
        self.leaky_rate = leaky_rate
        self.node_cnt = node_cnt
        self.dropout_rate = dropout_rate
        self.w_ks = nn.Linear(node_cnt, node_cnt)
        self.w_qs = nn.Linear(node_cnt, node_cnt)
        self.leakyrelu = nn.LeakyReLU(self.leaky_rate)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        k = self.w_ks(x)
        q = self.w_qs(x)

        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        return attn_weights


class LatentCorrelationLayer(nn.Module):
    def __init__(self, window, node_cnt, leaky_rate=0.1, dropout_rate=0.3):
        super().__init__()
        self.window = window
        self.node_cnt = node_cnt
        self.leaky_rate = leaky_rate
        self.dropout_rate = dropout_rate
        self.GRU = nn.GRU(self.window, self.node_cnt)
        self.GrapAttentionLayer = GraphAttentionLayer(
            self.leaky_rate, self.node_cnt, self.dropout_rate)

    def forward(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())

        input = input.permute(1, 0, 2).contiguous()
        attention = self.GrapAttentionLayer(input)
        attention = torch.mean(attention, dim=0)
        attention = 0.5 * (attention + attention.T)
        return attention


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h, adj)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(
            cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight',
                     'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
