import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import logging
import math
from core.config import configs

logger = logging.getLogger(__name__)


class LaplacianPyramid(nn.Module):

    def __init__(self, width, height, hierarchy=True):
        super(LaplacianPyramid, self).__init__()
        self.hierarchy = hierarchy
        self.layer1 = nn.Parameter(torch.Tensor(1, 1, width, height))
        if self.hierarchy:
            self.layer2 = nn.Parameter(torch.Tensor(1, 1, width // 2, height // 2))
            self.layer3 = nn.Parameter(torch.Tensor(1, 1, width // 4, height // 4))
            self.layer4 = nn.Parameter(torch.Tensor(1, 1, width // 8, height // 8))
        self.reset_parameters()

    def reset_parameters(self):
        # self.layer1.data.uniform_(0, 1)
        init.xavier_uniform_(self.layer1.data)
        if self.hierarchy:
            init.xavier_uniform_(self.layer2.data)
            init.xavier_uniform_(self.layer3.data)
            init.xavier_uniform_(self.layer4.data)

    def forward(self, x):
        """
        Args:
        - x: B × H × W × 2, UV coordinate, range [0, 1]

        Returns:
        - B × 1 × H × W 
        """
        batch = x.shape[0]
        x = x * 2.0 - 1.0       # normalize to [-1, 1] if you want to use grid_sample
        # about grid_sample, see more in https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        if self.hierarchy:
            y1 = F.grid_sample(self.layer1.repeat(batch, 1, 1, 1), x, align_corners=True)
            y2 = F.grid_sample(self.layer2.repeat(batch, 1, 1, 1), x, align_corners=True)
            y3 = F.grid_sample(self.layer3.repeat(batch, 1, 1, 1), x, align_corners=True)
            y4 = F.grid_sample(self.layer4.repeat(batch, 1, 1, 1), x, align_corners=True)
            y = y1 + y2 + y3 + y4
            return y
        else:
            y1 = F.grid_sample(self.layer1.repeat(batch, 1, 1, 1), x, align_corners=True)
            return y1


class Texture(nn.Module):

    def __init__(self, width, height, feature_num):
        super(Texture, self).__init__()
        self.width = width
        self.height = height
        self.feature_num = feature_num
        self.textures = nn.ModuleList([LaplacianPyramid(width, height, hierarchy=configs.NEURAL_TEXTURE.HIERARCHY) for i in range(feature_num)])
        # self.layer1 = nn.ParameterList()
        # for i in range(feature_num):
        #     self.layer1.append(self.textures[i].layer1)


    def forward(self, x):
        """
        Args:
        - x: B × H × W × 2

        Returns:
        - B × C × H × W
        """
        y_i = []
        for i in range(self.feature_num):
            y_i.append(self.textures[i](x))
        
        y = torch.cat(tuple(y_i), dim=1)
        return y


if __name__ == "__main__":
    model = Texture(128, 128, 12)

    params = list(model.named_parameters())
    print(params.__len__())
    print(params[0])
    print(params[1])
    print(params[2])
    print(params[3])
    print(params[4])

    # ipt = torch.rand((1, 128, 128, 2))
    # ouput = model(ipt)
    # print (ouput)