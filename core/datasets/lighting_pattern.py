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


class LightingPatternPool(Dataset):
    def __init__(self, root, light_num):
        self.root = root
        self.train_data = []        # 存储所有 lighting pattern 文件路径名
        self._init_data()
        self.light_num = light_num

    def _parse_path(self, path):
        """
        D:\\Code\\Project\\NeuralTexture_gan\\data\\gt\\0\\lighting_pattern\\0
        """
        path = path.replace('/', '\\')
        path = path.split('\\')
        folder = int(path[-3])
        image_idx = int(path[-1])
        return folder, image_idx

    def _init_data(self):
        """
        初始化 lighting pattern

        1_lp.pt   (384, )
        1_gt.pt   (3, 1400, 1400)
        """
        for i in range(6):
            self.train_data.append("{}\\gt\\0\\lighting_pattern\\{}".format(self.root, i))

    def __len__(self):
        return len(self.train_data)

    def get(self):
        """
        获取一个数据用于训练
        lighting_pattern: (384, )
        gt: (3, H, W)
        """
        path = random.choice(self.train_data)

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

        lp: (1, 384)

        计算出 gt, 然后把这两个都保存起来
        """

        num = len(self.train_data)

        lp_path = "{}\\gt\\0\\lighting_pattern\\{}_lp.pt".format(self.root, num)
        gt_path = "{}\\gt\\0\\lighting_pattern\\{}_gt.pt".format(self.root, num)
        
        gt = torch.zeros(3, 1400, 1400)

        lp = lp[0].to('cpu')
        lp = lp / math.sqrt(torch.sum(lp * lp)) # normilize

        for j in range(self.light_num):
            image_path = "D:\\Code\\Project\\NeuralTexture_gan\\data\\gt\\0\\img{:0>5d}_cam00.npy".format(j)
            image = to_torch(np.load(image_path))
            gt = gt + image * lp[j]
        
        torch.save(lp, lp_path)
        torch.save(gt, gt_path)

        self.train_data.append("{}\\gt\\0\\lighting_pattern\\{}".format(self.root, num))

        print("add new lighting pattern!!!")
        print(lp)