"""
    Modifications to the original J.Johnson's architecture:
        1. Instance normalization is used instead of batch normalization *
        2. Instead of learned up-sampling use nearest-neighbor up-sampling followed by convolution **
        3. No scaled tanh at the output of the network ***
    * Ulyanov showed that this gives better results, checkout the paper here: https://arxiv.org/pdf/1607.08022.pdf
    ** Distill pub blog showed this to have better results: http://distill.pub/2016/deconv-checkerboard/
    *** I tried using it even opened an issue on the original Johnson's repo (written in Lua) - no improvements
    Note: checkout the details about original Johnson's architecture here:
    https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
"""

import torch
import torch.nn as nn


class TransformerNet(nn.Module):
    def __init__(self, input_nc):
        super(TransformerNet, self).__init__()

        # Non-linearity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Down-sampling convolution layers
        self.conv1 = ConvLayer(input_nc, 32, kernel_size=7, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Up-sampling convolution layers
        self.up1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.up2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.up3 = ConvLayer(32, 3, kernel_size=7, stride=1)

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.up1(y)))
        y = self.relu(self.in5(self.up2(y)))
        y = self.up3(y)
        y = self.tanh(y)
        # No tanh activation here as originally proposed by J.Johnson, I didn't get any improvements by using it,
        # if you get better results using it feel free to make a PR
        return y


class ConvLayer(torch.nn.Module):
    """
        A small wrapper around nn.Conv2d, so as to make the code cleaner and allow for experimentation with padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        return self.conv2d(x)


class ResidualBlock(torch.nn.Module):
    """
        Originally introduced in (Microsoft Research Asia, He et al.): https://arxiv.org/abs/1512.03385
        Modified architecture according to suggestions in this blog: http://torch.ch/blog/2016/02/04/resnets.html
        The only difference from the original is: There is no ReLU layer after the addition of identity and residual
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual  # modification: no ReLu after the addition
        return out


class UpsampleConvLayer(torch.nn.Module):
    """
        Nearest-neighbor up-sampling followed by a convolution
        Appears to give better results than learned up-sampling aka transposed conv (avoids the checkerboard artifact)
        Initially proposed on distill pub: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.upsampling_factor = stride
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, stride=1)

    def forward(self, x):
        if self.upsampling_factor > 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsampling_factor, mode='nearest')
        return self.conv2d(x)