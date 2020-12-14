import torch.nn as nn
import torch
from .linear_net import LinearNet


class LPDiscriminator(nn.Module):
    def __init__(self, light_num, lp_dim, ndf):
        super(LPDiscriminator,self).__init__()

        self.linear = LinearNet(light_num, lp_dim)

        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(lp_dim + 3, ndf, kernel_size=4, stride=2, padding=1),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 32 x 32
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1),
                                 nn.BatchNorm2d(ndf*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 31 x 31
        self.layer5 = nn.Sequential(nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1),
                                 nn.Sigmoid()
                                 )
        # 30 x 30

    def forward(self, lp, x):
        """

        lp: lighting pattern (384, )
        x: 3 × H × W
        """
        H = x.size(2)
        W = x.size(3)

        out_lp = self.linear(lp)                                        # B × 6

        out_lp = out_lp.view(out_lp.size(0), out_lp.size(1), 1, 1)

        out_lp = out_lp.repeat(1, 1, H, W)    # B × 6 × H × W

        ipt = torch.cat((x, out_lp), 1)

        # print("ipt")
        # print (ipt)

        out1 = self.layer1(ipt)

        # print("out1")
        # print (out1)

        out2 = self.layer2(out1)

        # print("out2")
        # print (out2)

        out3 = self.layer3(out2)

        # print("out3")
        # print (out3)

        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return [out2, out3], out5

