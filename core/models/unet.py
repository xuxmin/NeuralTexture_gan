import torch
import torch.nn as nn


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, concat=False, final=False):
        super(up, self).__init__()
        self.concat = concat
        if final:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.Tanh()
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x, cat_x):
        if self.concat:
            x = torch.cat((cat_x, x), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        self.down1 = down(in_ch, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.up1 = up(512, 512)
        self.up2 = up(1024, 512, concat=True)                   # 512 + 512
        self.up3 = up(768, 256, concat=True)                    # 512 + 256
        self.up4 = up(384, 128, concat=True)                    # 256 + 128
        self.up5 = up(192, out_ch, concat=True, final=True)     # 128 + 64

    def forward(self, x):
        """
        x: B × C × H × W
        """
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, None)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        return x

if __name__ == "__main__":
    model = UNet(12, 3)

    params = list(model.named_parameters())
    print(params.__len__())
    print(params[0])
    print(params[1])
    print(params[2])
    print(params[3])