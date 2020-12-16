import torch
import torch.nn as nn
import logging
import math

from core.config import configs
from core.datasets.egg import EggDataset
from .texture import Texture
from .generator import Generator
from .linear_net import LinearNet

logger = logging.getLogger(__name__)


class LPGenerator(nn.Module):

    def __init__(self, width, height, feature_num, device):
        super(LPGenerator, self).__init__()

        self.lp_reduce_model = LinearNet(384, 6)                # 将 384 维降到 6 维

        self.feature_num = feature_num
        self.width = width
        self.height = height
        self.device = device

        self.texture = Texture(configs.NEURAL_TEXTURE.SIZE, configs.NEURAL_TEXTURE.SIZE, feature_num)

        # neural texture + lp + view_dir(3) + normal(3)
        self.generate_channels = feature_num + self.lp_reduce_model.out_dim + 6
        self.generator = Generator(self.generate_channels, 3, 64)

    def forward(self, uv_map, normal, view_dir, lp):
        """
        Args:
        - uv_map: B × 3 × H × W
        - mask: B × 3 × H × W
        - normal: B × 3 × H × W
        - view_dir: B × 3
        - lp: B × 384
        """
        uv_map = uv_map.permute(0, 2, 3, 1)[:, :, :, :2]                            # B × H × W × 2

        x = self.texture(uv_map)                                                    # B × C × H × W

        view_dir = view_dir.view(view_dir.shape[0], view_dir.shape[1], 1, 1)        # B × 3 × H × W
        view_dir = view_dir.repeat(1, 1, self.height, self.width)
        view_dir = view_dir.type(torch.cuda.FloatTensor)

        lp = self.lp_reduce_model(lp)                                                        # B × 6
        lp = lp.view(lp.shape[0], lp.shape[1], 1, 1)                                # B × 6 × H × W
        lp = lp.repeat(1, 1, self.height, self.width)
        lp = lp.type(torch.cuda.FloatTensor)

        x = torch.cat(tuple([x, normal, view_dir, lp]), dim=1)

        y = self.generator(x)

        return x, y




"""
class LPGenerator(nn.Module):

    def __init__(self, width, height, feature_num, device):
        super(LPGenerator, self).__init__()
        self.feature_num = feature_num
        self.width = width
        self.height = height
        self.device = device

        self.pipeline = NewGanPipelineModel(configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1], configs.NEURAL_TEXTURE.FEATURE_NUM, device).to(device)
        self.eggDataset = EggDataset(root=configs.DATASET.ROOT, is_train=True)

    def generate(self, uv_map, normal, view_dir, light_dir):
        real_A, fake_B = self.pipeline(uv_map, normal, view_dir, light_dir)
        return real_A, fake_B


    def forward(self, lp):

        egg_loader = torch.utils.data.DataLoader(self.eggDataset, batch_size=64, shuffle=False, pin_memory=True)
    
        pred = torch.zeros((1, 3, self.height, self.width)).to(self.device)

        result = []

        for i, (uv_map, gt, masks, normal, light_dir, view_dir) in enumerate(egg_loader):

            gt = torch.log(math.exp(-3)+gt) / 3
        
            uv_map = uv_map.to(self.device)
            gt = gt.to(self.device)
            normal = normal.to(self.device)
            light_dir = light_dir.to(self.device)
            view_dir = view_dir.to(self.device)

            _, fake_B = self.pipeline(uv_map, normal, view_dir, light_dir)

            fake_B = torch.exp(fake_B * 3) - math.exp(-3)       # 变换到 [0, ∞], B × 3 × H × W

            # pred = pred + fake_B * lp[0, i]

            result.append(fake_B)
            
        pred = torch.cat(tuple(result), 0)                      # 384 × 3 × H × W

        lp = lp.unsqueeze(0).view(-1, 1, 1, 1)

        pred = pred * lp                                        # lp: 384 × 1 × 1 × 1

        pred = torch.sum(pred, 0, keepdim=True)                 # 3 × H × W

        pred = torch.log(math.exp(-3) + pred) / 3               # pred range [-∞, +∞]

        return pred
"""
