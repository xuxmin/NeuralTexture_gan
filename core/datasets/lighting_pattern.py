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
from core.utils.osutils import join
from core.utils.osutils import exists
from core.utils.osutils import mkdir_p
from core.utils.misc import normalize

logger = logging.getLogger(__name__)

"""
TODO:
- 支持多个 view
"""


class LightingPatternPool(Dataset):
    def __init__(self, root, lp_num, checkpoint_dir):
        """
        root: 初始 lighting pattern 所在路径, 例如 D:\\Code\\Project\\NeuralTexture_gan\\data\\lighting_pattern\\32
        lp_num: 当前阶段的 lighting pattern 数目(就是训练了一段时间后的)
        """
        self.root = root
        self.initial_dir = join(root, 'lighting_pattern')                  # 初始 lighting pattern 存放位置
        self.initial_lp_num = configs.LIGHTING_PATTERN.INITIAL_NUM              # 初始 lighting pattern 数目
        self.checkpoint_dir = join(checkpoint_dir, 'lighting_pattern')          # 生成的 lighting pattern 存放位置
        if not exists(self.checkpoint_dir):
            mkdir_p(self.checkpoint_dir)

        self.train_data = []                # 存储所有 lighting pattern 文件路径名
        self._init_data(lp_num)             # 初始化已有的 lighting pattern
        self.cal_weight(lp_num)             # 计算每个 lighting pattern 的概率

    def _parse_path(self, path):
        """
        ..\\lighting_pattern\\0\\0
        """
        path = path.replace('/', '\\')
        path = path.split('\\')
        folder = int(path[-2])
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

        初始数量设置为 x, 每个等概率为 1/x, 每次新增两个 lighting pattern
        lp_num x+2: 最后两个 2/(x+2), 其余 (x-2)/[x(x+2)]
        lp_num x+4: 最后两个 3/(x+4), 其余 (x-2)/[(x+2)(x+4)]
        """
        assert(lp_num % 2 == 0)
        
        self.weight = torch.zeros(lp_num)

        for i in range(lp_num-2):
            self.weight[i] = (self.initial_lp_num - 2) / (lp_num * (lp_num - 2))
        
        self.weight[-1] = self.weight[-2] = (1 + (lp_num - self.initial_lp_num) // 2) / lp_num


    def _init_data(self, lp_num):
        """
        初始化 lighting pattern

        1_lp.pt   (384, )
        1_gt.pt   (3, 1400, 1400)
        """
        # 把初始的 lighting pattern 加进去
        for i in range(self.initial_lp_num):
            self.train_data.append(join(self.initial_dir, '0', str(i)))

        # 把训练生成的 lighting pattern(在 checkpoint_dir 里面), 也加进去
        for i in range(self.initial_lp_num, lp_num):
            self.train_data.append(join(self.checkpoint_dir, '0', str(i)))
        
        print

    def __len__(self):
        return len(self.train_data)

    def get(self, index=None):
        """
        如果不提供 index, 则以 weight 为概率随机获取, 否则按照 index 下标获取

        获取一个数据用于训练
        lighting_pattern: (384, )
        gt: (3, H, W)
        """
        # 根据 weight 随机选取数据
        if not index:
            path = random.choices(self.train_data, self.weight)[0]
        eles:
            path = self.train_data[index]

        folder_idx, image_idx = self._parse_path(path)              # folder_idx 表示不同的 view, image_idx 对应 lighting pattern
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
        lighting_pattern = normalize(lighting_pattern)

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
        VIEW = '0'

        logger.info("gradient: {}".format(lp))
        lp = lp[0].to('cpu')

        num = len(self.train_data)              # lighting pattern 的数目

        # 正交化
        # for i in range(num):
        #     path = self.train_data[i]
        #     lp_i = torch.load("{}_lp.pt".format(path))
        #     lp_i = lp_i / math.sqrt(torch.sum(lp_i * lp_i))     # normilize
        #     lp = lp - torch.sum(lp * lp_i) * lp_i
        
        lp_plus = lp.clone()
        lp_minus = lp.clone()

        # 分离梯度值的正值与负值
        lp_plus[lp < 0] = 0
        lp_minus[lp > 0] = 0
        lp_minus = torch.abs(lp_minus)

        # normalize
        if torch.sum(lp_plus * lp_plus) != 0:
            lp_plus = normalize(lp_plus)

        if torch.sum(lp_minus * lp_minus) != 0:
            lp_minus = normalize(lp_minus)
        
        if torch.sum(lp_plus * lp_plus) < 1e-6:
            lp_plus = lp_minus
        if torch.sum(lp_minus * lp_minus) < 1e-6:
            lp_minus = lp_plus

        gt_plus = torch.zeros(3, 1400, 1400)
        gt_minus = torch.zeros(3, 1400, 1400)

        for j in range(384):
            image_path = "{}\\gt\\{}\\img{:0>5d}_cam00.npy".format(self.root, VIEW, j)
            image = to_torch(np.load(image_path))
            gt_plus = gt_plus + image * lp_plus[j]
            gt_minus = gt_minus + image * lp_minus[j]
        
        if not exists(join(self.checkpoint_dir, VIEW)):
            mkdir_p(join(self.checkpoint_dir, VIEW))

        lp_plus_path = join(self.checkpoint_dir, VIEW, "{}_lp.pt".format(num))
        gt_plus_path = join(self.checkpoint_dir, VIEW, "{}_gt.pt".format(num))
        lp_minus_path = join(self.checkpoint_dir, VIEW,  "{}_lp.pt".format(num+1))
        gt_minus_path = join(self.checkpoint_dir, VIEW, "{}_gt.pt".format(num+1))

        torch.save(lp_plus, lp_plus_path)
        torch.save(gt_plus, gt_plus_path)
        torch.save(lp_minus, lp_minus_path)
        torch.save(gt_minus, gt_minus_path)

        self.train_data.append(join(self.checkpoint_dir, VIEW, str(num)))
        self.train_data.append(join(self.checkpoint_dir, VIEW, str(num+1)))

        logger.info("add new lighting pattern!!!")
        logger.info(lp_plus)
        logger.info(lp_minus)

        # 每次增加两个新的 lighting pattern 后, 应该重新计算 weight
        self.cal_weight(len(self.train_data))
        logger.info(self.weight)