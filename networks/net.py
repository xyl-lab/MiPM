from networks.layer import *
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class gtnet(nn.Module, BaseEstimator):
    def __init__(self, device_name='cuda:0', gcn_depth=2, propalpha=0.05, leaky_rate=0.3, dropout=0.3):
        super(gtnet, self).__init__()
        self.device_name = device_name
        self.device = torch.device(self.device_name)
        # whether to add graph convolution layer
        # whether to construct adaptive adjacency matrix
        # graph convolution depth
        self.gcn_depth = gcn_depth
        # number of nodes/variables
        self.num_nodes = 45
        # dim of nodes
        self.node_dim = 40
        # prop alpha
        self.propalpha = propalpha
        # tanh alpha
        self.tanhalpha = 3
        # dilation exponential卷积层膨胀系数的增长速率
        self.dilation_exponential = 2
        # number of layers
        self.layers = 3
        # node neighbor
        self.out_channels = 64
        self.dropout = dropout
        self.leaky_rate = leaky_rate
        self.conv_channels = self.out_channels
        self.residual_channels = self.out_channels

        self.in_dim = 1

        self.skip_channels = 32
        self.end_channels = 64
        self.skip_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        self.latentcorrelationlayer = LatentCorrelationLayer(
            64, self.num_nodes)
        # input sequence length
        self.seq_length = 64
        # output sequence length
        self.out_dim = 1
        self.layer_norm_affline = False
        kernel_size = 7
        self.receptive_field = int(
            1+(kernel_size-1)*(self.dilation_exponential**self.layers-1))
        new_dilation = 1
        for j in range(1, self.layers+1):
            # padding=0
            rf_size_j = int(
                1 + (kernel_size-1)*(self.dilation_exponential**j-1))
            self.gate_convs.append(dilated_1D(
                self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
            if self.seq_length > self.receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, self.seq_length-rf_size_j+1)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, self.receptive_field-rf_size_j+1)))
            self.gconv1.append(
                mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout, self.propalpha))
            self.gconv2.append(
                mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout, self.propalpha))

            if self.seq_length > self.receptive_field:
                self.norm.append(LayerNorm(
                    (self.residual_channels, self.num_nodes, self.seq_length - rf_size_j + 1), elementwise_affine=self.layer_norm_affline))
            else:
                self.norm.append(LayerNorm(
                    (self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1), elementwise_affine=self.layer_norm_affline))

            new_dilation *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(
                1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(
                1, self.seq_length-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(
                1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(self.device)

    def forward(self, x, y):
        input = torch.cat((x, y), dim=-2)
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(
                input, (self.receptive_field-self.seq_length, 0, 0, 0))
        tmp = input.squeeze(1)
        tmp = tmp.transpose(-2, -1)
        adp = self.latentcorrelationlayer(tmp)
        x = self.start_conv(input)
        skip = self.skip0(
            F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            # TC Module
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = gate
            x = F.dropout(x, self.dropout, training=self.training)
            # Skip Connection
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            # GC Module
            x = self.gconv1[i](x, adp)+self.gconv2[i](x,
                                                      adp.transpose(1, 0))
            # add & Norm
            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx)
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x[:, :, -7:, :]
        return x, None
