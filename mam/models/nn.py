import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mar_utils_mol import convert_val_to_onehot

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class MLP(nn.Module):
    def __init__(self, layers, K, with_ln=False, act=nn.LeakyReLU(), tail=[]):
        super().__init__()
        self.K = K
        if with_ln:
            self.net = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([nn.LayerNorm(o), act] if n < len(layers) - 2 else [])
                for n, (i, o) in enumerate(zip(layers, layers[1:]))], []
            ) + tail))
        else:
            self.net = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([act] if n < len(layers) - 2 else [])
                for n, (i, o) in enumerate(zip(layers, layers[1:]))], []
            ) + tail))

    def forward(self, x):
        x = convert_val_to_onehot(x, self.K)
        x = x.flatten(-2,-1)
        return self.net(x).squeeze(-1) # logp

class MLPRes(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, res=True):
        super(MLPRes, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers-1)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU()
        self.res = res

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = layer(x)
            out = self.activation(out)
            if self.res:
                out = out + x  # Residual connection
            x = out
        out = self.output_layer(x)
        return out
    
class MLPResDual(nn.Module):
    def __init__(self, hidden_dim, K, L, num_layers, res=True):
        super(MLPResDual, self).__init__()
        input_dim = (K + 1) * L
        num_logits = K * L
        self.mlp_ar = MLPRes(input_dim, hidden_dim, num_logits, num_layers, res)
        self.mlp_marg = MLPRes(input_dim, hidden_dim, 1, num_layers, res)
        self.K = K
        self.L = L

    def forward(self, x):
        x = convert_val_to_onehot(x, self.K)
        x = x.flatten(-2,-1)
        logits_ar = self.mlp_ar(x)
        logp = self.mlp_marg(x)
        return logp.squeeze(-1), logits_ar.reshape(x.shape[0], self.L, self.K) # logp, logits
    
class MLPResSingle(nn.Module):
    def __init__(self, hidden_dim, K, L, num_layers, res=True):
        super(MLPResSingle, self).__init__()
        input_dim = (K + 1) * L
        num_logits = K * L
        self.mlp_ar = MLPRes(input_dim, hidden_dim, num_logits, num_layers, res)
        self.K = K
        self.L = L

    def forward(self, x):
        x = convert_val_to_onehot(x, self.K)
        x = x.flatten(-2,-1)
        logits_ar = self.mlp_ar(x)
        return logits_ar.reshape(x.shape[0], self.L, self.K) # logp, logits
    
class MLPARDual(nn.Module):
    def __init__(self, layers, K, L, with_ln=False, act=nn.LeakyReLU(), tail=[]):
        super().__init__()
        self.K = K
        self.L = L
        layers_logp = layers + [1]
        layers_ar = layers + [L*K]
        if with_ln:
            self.net_logp = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([nn.LayerNorm(o), act] if n < len(layers) - 1 else [])
                for n, (i, o) in enumerate(zip(layers_logp, layers_logp[1:]))], []
            ) + tail))
            self.net_ar = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([nn.LayerNorm(o), act] if n < len(layers) - 1 else [])
                for n, (i, o) in enumerate(zip(layers_ar, layers_ar[1:]))], []
            ) + tail))
        else:
            self.net_logp = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([act] if n < len(layers) - 1 else [])
                for n, (i, o) in enumerate(zip(layers_logp, layers_logp[1:]))], []
            ) + tail))
            self.net_ar = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([act] if n < len(layers) - 1 else [])
                for n, (i, o) in enumerate(zip(layers_ar, layers_ar[1:]))], []
            ) + tail))

    def forward(self, x):
        x = convert_val_to_onehot(x, self.K)
        x = x.flatten(-2,-1)
        logp = self.net_logp(x)
        logits_ar = self.net_ar(x)
        return logp.squeeze(-1), logits_ar.reshape(x.shape[0], self.L, self.K) # logp, logits

class MLPAR(nn.Module):
    def __init__(self, layers, K, L, with_ln=False, act=nn.LeakyReLU(), tail=[]):
        super().__init__()
        self.K = K
        self.L = L
        layers = layers + [1 + L*K]
        if with_ln:
            self.net = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([nn.LayerNorm(o), act] if n < len(layers) - 2 else [])
                for n, (i, o) in enumerate(zip(layers, layers[1:]))], []
            ) + tail))
        else:
            self.net = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([act] if n < len(layers) - 2 else [])
                for n, (i, o) in enumerate(zip(layers, layers[1:]))], []
            ) + tail))

    def forward(self, x):
        x = convert_val_to_onehot(x, self.K)
        x = x.flatten(-2,-1)
        out = self.net(x)
        return out[:,0], out[:,1:].reshape(x.shape[0], self.L, self.K) # logp, logits

class MLPAR_FB(nn.Module):
    def __init__(self, layers, K, L, with_ln=False, act=nn.LeakyReLU(), tail=[]):
        super().__init__()
        self.K = K
        self.L = L
        self.logits_size = self.L*self.K
        if with_ln:
            self.net = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([nn.LayerNorm(o), act] if n < len(layers) - 2 else [])
                for n, (i, o) in enumerate(zip(layers, layers[1:]))], []
            ) + tail))
        else:
            self.net = nn.Sequential(*(sum(
                [[nn.Linear(i, o)] + ([act] if n < len(layers) - 2 else [])
                for n, (i, o) in enumerate(zip(layers, layers[1:]))], []
            ) + tail))

    def forward(self, x):
        x = convert_val_to_onehot(x, self.K)
        x = x.flatten(-2,-1)
        out = self.net(x) # of size (B, L*(K+1))
        add_logits = out[:,:self.logits_size] # of size (B, L*K)
        del_logits = out[:,self.logits_size:] # of size (B, L)
        return add_logits, del_logits

def make_mlp(l, act=nn.LeakyReLU(), tail=[], with_ln=False):
    """makes an MLP with no top layer activation"""
    if with_ln:
        net = nn.Sequential(*(sum(
            [[nn.Linear(i, o)] + ([nn.LayerNorm(o), act] if n < len(l) - 2 else [])
            for n, (i, o) in enumerate(zip(l, l[1:]))], []
        ) + tail))
    else:
        net = nn.Sequential(*(sum(
            [[nn.Linear(i, o)] + ([act] if n < len(l) - 2 else [])
            for n, (i, o) in enumerate(zip(l, l[1:]))], []
        ) + tail))
    return net

def mlp_ebm(nin, nint=256, nout=1, act=nn.LeakyReLU()):
    return nn.Sequential(
        nn.Linear(nin, nint),
        act,
        nn.Linear(nint, nint),
        act,
        nn.Linear(nint, nint),
        act,
        nn.Linear(nint, nout),
    )

def conv_transpose_3x3(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=1, bias=True)

def conv3x3(in_planes, out_planes, stride=1):
    if stride < 0:
        return conv_transpose_3x3(in_planes, out_planes, stride=-stride)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, act = Swish(), stride=1, layer_norm=False, out_nonlin=True):
        super(BasicBlock, self).__init__()
        self.nonlin1 = act
        self.nonlin2 = act
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        if layer_norm:
            raise NotImplementedError("LayerNorm not implemented")
        self.out_nonlin = out_nonlin
        self.shortcut_conv = None
        if stride != 1 or in_planes != self.expansion * planes:
            if stride < 0:
                self.shortcut_conv = nn.ConvTranspose2d(in_planes, self.expansion*planes,
                                                        kernel_size=1, stride=-stride,
                                                        output_padding=1, bias=True)
            else:
                self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes,
                                               kernel_size=1, stride=stride, bias=True)
                                               
    def forward(self, x):
        out = self.conv1(x)
        out = self.nonlin1(out)
        out = self.conv2(out)
        if self.shortcut_conv is not None:
            out_sc = self.shortcut_conv(x)
            out += out_sc
        else:
            out += x
        if self.out_nonlin:
            out = self.nonlin2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, n_channels=64, n_hids=[2048,1], act=Swish(), encoding="onehot", is_downsample=False):
        super().__init__()
        self.encoding = encoding
        if self.encoding == "onehot":
            self.proj = nn.Conv2d(3, n_channels, 3, 1, 1)
        elif self.encoding == "2bit":
            self.proj = nn.Conv2d(2, n_channels, 3, 1, 1)
        elif self.encoding == "bit":
            self.proj = nn.Conv2d(1, n_channels, 3, 1, 1)
        downsample = [
            BasicBlock(n_channels, n_channels, act=act, stride=2),
            BasicBlock(n_channels, n_channels, act=act, stride=2),
        ]
        main = [
            BasicBlock(
            n_channels, n_channels, act=act, stride=1
            ) for _ in range(6)
        ]
        if is_downsample:
            all = downsample + main
        else:
            all = main
        self.net = nn.Sequential(*all)
        self.flatten = nn.Flatten(-3, -1)
        linear_layers = [[nn.LazyLinear(n_hid)] + ([act] if n < len(n_hids) - 1 else [])
            for n, (n_hid) in enumerate(n_hids)]
        self.mlp = nn.Sequential(*(sum(linear_layers, [])))

    def forward(self, input):
        batch_dim = input.shape[0:-1]
        if self.encoding == "onehot":
            input = input.view(-1, 28, 28, 3).permute(0, 3, 1, 2) # (B, 3, 28, 28)
        elif self.encoding == "2bit":
            input = input.view(-1, 28, 28, 2).permute(0, 3, 1, 2) # (B, 2, 28, 28)
        elif self.encoding == "bit":
            input = input.view(-1, 28, 28, 1).permute(0, 3, 1, 2) # (B, 1, 28, 28)

        input = self.proj(input)
        out = self.net(input)
        out = self.flatten(out)
        out = self.mlp(out)
        out = out.view(batch_dim)
        return out

class MNVit(nn.Module):
    def __init__(self, net, encoding):
        super().__init__()
        self.encoding = encoding
        self.net = net
    def forward(self, x):
        batch_dim = x.shape[0:-1]
        if self.encoding == "onehot":
            x = x.view(-1, 28, 28, 3).permute(0, 3, 1, 2) # (B, 3, 28, 28)
        elif self.encoding == "2bit":
            x = x.view(-1, 28, 28, 2).permute(0, 3, 1, 2) # (B, 2, 28, 28)
        elif self.encoding == "bit":
            x = x.view(-1, 28, 28, 1).permute(0, 3, 1, 2) # (B, 1, 28, 28)
        output = self.net(x).squeeze()
        output = output.view(batch_dim)
        return output

class ConvNet(nn.Module):
    def __init__(self, n_channels=[16]*2, n_hids=[120, 84, 1], kernel_sizes=[5]*2, act=nn.Sigmoid(), 
        pool=None, norm2d=None, norm1d=None, encoding="onehot"):
        super().__init__()
        if norm1d is not None:
            norm_1d_list = [norm1d(n_hid) for n_hid in n_hids[:-1]]
        conv_layers = [[nn.LazyConv2d(n_ch, kernel_size=kernel_size)] + \
            ([norm2d(n_ch, n_ch)] if norm2d else []) + [act] + ([pool] if pool else [])
            for n_ch, kernel_size in zip(n_channels,kernel_sizes)]
        linear_layers = [[nn.LazyLinear(n_hid)] + ((([norm_1d_list[n]] if norm1d else []) + [act]) if n < len(n_hids) - 1 else [])
            for n, (n_hid) in enumerate(n_hids)]
        layers = sum(conv_layers, []) + [nn.Flatten(-3, -1)] + sum(linear_layers, [])
        self.net = nn.Sequential(*layers)
        self.encoding = encoding
    def forward(self, x):
        batch_dim = x.shape[0:-1]
        if self.encoding == "onehot":
            x = x.view(-1, 28, 28, 3).permute(0, 3, 1, 2) # (B, 3, 28, 28)
        elif self.encoding == "2bit":
            x = x.view(-1, 28, 28, 2).permute(0, 3, 1, 2) # (B, 2, 28, 28)
        elif self.encoding == "bit":
            x = x.view(-1, 28, 28, 1).permute(0, 3, 1, 2) # (B, 1, 28, 28)
        output = self.net(x).squeeze()
        output = output.view(batch_dim)
        return output

def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)