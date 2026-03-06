import math

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm



class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, use_maxpool=False):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        if use_maxpool and stride > 1:
            raise ValueError("Cannot use both Conv1d with stride > 1 and MaxPool1d. Choose one method for downsampling.")

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.use_maxpool:
            x = self.pool(x)
        return x


class Deconv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, output_padding=0, use_upsample=False):
        super(Deconv1DBlock, self).__init__()
        # ConvTranspose1d upsamples the sequence length
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.use_upsample = use_upsample
        if use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        if use_upsample and stride > 1:
            raise ValueError("Cannot use both ConvTranspose1d with stride > 1 and Upsample. Choose one method for upsampling.")

    def forward(self, x):
        x = self.relu(self.bn(self.deconv(x)))
        if self.use_upsample:
            x = self.upsample(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, use_maxpool=False):
        super(Conv2DBlock, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.LeakyReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if use_maxpool and stride > 1:
            raise ValueError("Cannot use both Conv2d with stride > 1 and MaxPool2d. Choose one method for downsampling.")

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.use_maxpool:
            x = self.pool(x)
        return x

class Deconv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=0, use_upsample=False):
        super(Deconv2DBlock, self).__init__()
        # ConvTranspose2d upsamples the spatial dimensions
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.use_upsample = use_upsample
        if use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        if use_upsample and stride > 1:
            raise ValueError("Cannot use both ConvTranspose2d with stride > 1 and Upsample. Choose one method for upsampling.")

    def forward(self, x):
        x = self.relu(self.bn(self.deconv(x)))
        if self.use_upsample:
            x = self.upsample(x)
        return x

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, use_maxpool=False):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        if use_maxpool and stride > 1:
            raise ValueError("Cannot use both Conv3d with stride > 1 and MaxPool3d. Choose one method for downsampling.")

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.use_maxpool:
            x = self.pool(x)
        return x

class Deconv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=0, use_upsample=False):
        super(Deconv3DBlock, self).__init__()
        # ConvTranspose3d upsamples the spatial dimensions
        self.deconv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()
        self.use_upsample = use_upsample
        if use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        if use_upsample and stride > 1:
            raise ValueError("Cannot use both ConvTranspose3d with stride > 1 and Upsample. Choose one method for upsampling.")

    def forward(self, x):
        x = self.relu(self.bn(self.deconv(x)))
        if self.use_upsample:
            x = self.upsample(x)
        return x



