import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import torch_dct as dct
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ResidualBlock_nopading(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_nopading, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=12):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, \
            stride=1, padding=1, dilation=1),
            nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Residual blocks USV
        res_blocks_USV = []
        for _ in range(n_residual_blocks):
            res_blocks_USV.append(ResidualBlock(3))
        self.res_blocks_USV = nn.Sequential(*res_blocks_USV)

        
        self.U = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, \
            stride=1, padding=1, dilation=1),
            nn.PReLU())

        self.S = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, \
            stride=1, padding=1, dilation=1),
            nn.PReLU(),
            # nn.AdaptiveAvgPool2d(8),
            nn.Flatten(2),
            nn.Linear(1024,32))
        
        
        # self.S = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, \
        #     stride=1, padding=1, dilation=1),
        #     nn.PReLU())

        
        self.V = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, \
            stride=1, padding=1, dilation=1),
            nn.PReLU())

        res_blocks_nopadding = []
        for _ in range(n_residual_blocks):
            res_blocks_nopadding.append(ResidualBlock_nopading(64))
        self.res_blocks_nopadding = nn.Sequential(*res_blocks_nopadding)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, \
            padding=0, dilation=1),
            nn.BatchNorm2d(64, 0.8))

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(3, out_channels, kernel_size=1, \
            stride=1, padding=0, dilation=1), nn.ReLU(), nn.Sigmoid())

    def forward(self, x):

        x = self.res_blocks_USV(x)
        
        U = self.U(x)
        V = self.V(x)
        S = self.S(x)
        
        x = torch.matmul(torch.matmul(U, torch.diag_embed(S)),V.transpose(-2,-1))
            
        
        # out1 = self.conv1(x)

        # out1_ = self.res_blocks(out1)
       
        # out2 = self.conv2(out1_)

        # out2 = self.res_blocks_nopadding(out2)

        # out = torch.cat((out1_, out2), dim=1)

        out = self.conv3(x)

        return out

class CombineModel(nn.Module):

    def __init__(self, modelA, modelB):
        super(CombineModel, self).__init__()

        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        z = self.modelA(x)
        out = self.modelB(z)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)