
import math
import torch
import skimage.feature


def accuracy(output, target, mask):
    '''

    Args:
    - output(tensor(NxCxWxH)): 模型的输出
    - target(tensor(NxCxWxH)): 正确的图片
    - masks(tensor(NxCxWxH)): 

    Returns:
    '''
    criterionL1 = torch.nn.L1Loss(reduction='mean')
    criterionL2 = torch.nn.MSELoss(reduce = True)

    masks = (mask == 1)
    # mask_output = torch.masked_select(output, masks)
    # mask_target = torch.masked_select(target, masks)
    mask_output = output
    mask_target = target
    MAE = criterionL1(mask_output, mask_target)             # 平均绝对距离

    MSE = criterionL2(mask_output, mask_target)

    return MAE, MSE


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        val: n样本的值
        n: 样本的数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
