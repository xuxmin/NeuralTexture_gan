import torch
import torch.nn as nn
import logging
import math

from core.config import configs
from core.datasets.egg import EggDataset
from .pipeline import NewGanPipelineModel

logger = logging.getLogger(__name__)


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
        """

        lp: B × 384 已经归一化后的
        """
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

        # pred = torch.log(math.exp(-3) + pred) / 3             # pred range [-∞, +∞]

        return pred

