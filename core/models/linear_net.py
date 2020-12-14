import torch
import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(512, 100), 
            # nn.BatchNorm1d(100), 
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(100, out_dim)
        )
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
