import random
import logging
import torch
import math
import numpy as np
from torch.utils.data import Dataset

from core.config import configs
from core.utils.camutils import load_bin
from core.utils.imutils import load_png
from core.utils.imutils import resize
from core.utils.imutils import to_torch
from core.utils.osutils import isfile

logger = logging.getLogger(__name__)

"""
初始 lighting pattern 数目为 36, 不然得改代码...
"""


class LightingPatternPool(Dataset):
    def __init__(self, root, lp_num):
        self.root = root
        self.train_data = []        # 存储所有 lighting pattern 文件路径名
        self._init_data(lp_num)           # 初始化 lighting pattern
        self.cal_weight(lp_num)           # 计算每个 lighting pattern 的概率

    def _parse_path(self, path):
        """
        D:\\Code\\Project\\NeuralTexture_gan\\data\\gt\\0\\lighting_pattern\\0
        """
        path = path.replace('/', '\\')
        path = path.split('\\')
        folder = int(path[-3])
        image_idx = int(path[-1])
        return folder, image_idx
    
    def cal_weight(self, lp_num):
        """
        根据当前的 lighting pattern 的数目, 以及初始的 lighting pattern 数量
        可以计算出每个 lighting pattern 选取的概率, 保存在 weight 中

        初始数量设置为 36 个的话, 每个等概率 1/36
        lp_num 37: 最后一个为 2/37, 其余 35/(36*37)
        lp_num 38: 最后一个为 3/38, 其余 35/(37*38)
        lp_num x: 最后一个为 (x-35)/x, 其余为 35/(x*(x-1))

        如果是每次增加两个新的 lighting pattern...
        lp_num 38: 最后两个为 2/38, 其余 34/(36*38)
        lp_num 40: 最后两个为 3/40, 其余 34/(38*40)
        """
        assert(lp_num % 2 == 0)
        
        self.weight = torch.zeros(lp_num)

        for i in range(lp_num-2):
            self.weight[i] = 34 / (lp_num * (lp_num - 2))
        
        self.weight[-1] = (1 + (lp_num - 36) // 2) / lp_num
        self.weight[-2] = (1 + (lp_num - 36) // 2) / lp_num


    def _init_data(self, lp_num):
        """
        初始化 lighting pattern

        1_lp.pt   (384, )
        1_gt.pt   (3, 1400, 1400)
        """
        for i in range(lp_num):
            self.train_data.append("{}\\gt\\0\\lighting_pattern\\{}".format(self.root, i))

    def __len__(self):
        return len(self.train_data)

    def get(self):
        """
        获取一个数据用于训练
        lighting_pattern: (384, )
        gt: (3, H, W)
        """
        # 随机等概率选取数据
        path = random.choices(self.train_data, self.weight)[0]

        folder_idx, image_idx = self._parse_path(path)
        mask_path =  self.root + "\\gt\\{}\\mask_cam00.png".format(folder_idx)

        # load mask
        mask_path_tmp = mask_path.replace('.png', '.pt')
        if isfile(mask_path_tmp):
            mask = torch.load(mask_path_tmp)
        else:
            mask = load_png(mask_path)
            torch.save(mask, mask_path_tmp)
        mask = mask[:, 1100:2500, 1300:2700]

        # load lighting pattern
        lighting_pattern = torch.load("{}_lp.pt".format(path))
        lighting_pattern = lighting_pattern / math.sqrt(torch.sum(lighting_pattern * lighting_pattern)) # normilize

        # load gt
        gt = torch.load("{}_gt.pt".format(path))
        gt[mask == 0] = 0

        # resize mask, gt
        gt = resize(gt, configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1])
        mask = resize(mask, configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1])

        return lighting_pattern, gt, mask

    def __getitem__(self, index):
        return self.get()

    def add(self, lp):
        """
        增加一个新的 lighting pattern

        lp: (1, 384)    经过梯度下降获得的梯度值!!! 注意是梯度值

        计算出 gt, 然后把这两个都保存起来
        """
        logger.info("gradient: {}".format(lp))
        
        lp = lp[0].to('cpu')
        lp_plus = lp.clone()
        lp_minus = lp.clone()

        lp_plus[lp < 0] = 0
        lp_minus[lp > 0] = 0
        lp_minus = torch.abs(lp_minus)

        # normalize
        if torch.sum(lp_plus * lp_plus) != 0:
            lp_plus = lp_plus / math.sqrt(torch.sum(lp_plus * lp_plus))

        if torch.sum(lp_minus * lp_minus) != 0:
            lp_minus = lp_minus / math.sqrt(torch.sum(lp_minus * lp_minus))
        
        if torch.sum(lp_plus * lp_plus) < 1e-6:
            lp_plus = lp_minus
        if torch.sum(lp_minus * lp_minus) < 1e-6:
            lp_minus = lp_plus

        gt_plus = torch.zeros(3, 1400, 1400)
        gt_minus = torch.zeros(3, 1400, 1400)

        for j in range(384):
            image_path = "{}\\gt\\0\\img{:0>5d}_cam00.npy".format(self.root, j)
            image = to_torch(np.load(image_path))
            gt_plus = gt_plus + image * lp_plus[j]
            gt_minus = gt_minus + image * lp_minus[j]

        num = len(self.train_data)

        lp_plus_path = "{}\\gt\\0\\lighting_pattern\\{}_lp.pt".format(self.root, num)
        gt_plus_path = "{}\\gt\\0\\lighting_pattern\\{}_gt.pt".format(self.root, num)
        lp_minus_path = "{}\\gt\\0\\lighting_pattern\\{}_lp.pt".format(self.root, num+1)
        gt_minus_path = "{}\\gt\\0\\lighting_pattern\\{}_gt.pt".format(self.root, num+1)

        torch.save(lp_plus, lp_plus_path)
        torch.save(gt_plus, gt_plus_path)
        torch.save(lp_minus, lp_minus_path)
        torch.save(gt_minus, gt_minus_path)

        self.train_data.append("{}\\gt\\0\\lighting_pattern\\{}".format(self.root, num))
        self.train_data.append("{}\\gt\\0\\lighting_pattern\\{}".format(self.root, num+1))

        logger.info("add new lighting pattern!!!")
        logger.info(lp_plus)
        logger.info(lp_minus)

        # 每次增加两个新的 lighting pattern 后, 应该重新计算 weight
        self.cal_weight(len(self.train_data))
        logger.info(self.weight)