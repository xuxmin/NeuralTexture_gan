import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import math
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from core.models.pipeline import NewGanPipelineModel
from core.datasets.egg import EggDataset

from core.config import configs
from core.config import update_config
from core.utils.misc import create_logger
from core.utils.osutils import join
from core.utils.osutils import isfile
from core.utils.imutils import show_img_list
from core.utils.imutils import CEToneMapping
from core.utils.evaluation import AverageMeter
from core.utils.evaluation import accuracy

# cfg = "experiments/final/newganpipeline_tex256_f16_alldata_loss1-10-10-10.yaml"
# cfg = "experiments/final/newganpipeline_tex256_f16_alldata_loss0-1-0-0.yaml"
cfg = "experiments/final/newganpipeline_tex256_f16_alldata_loss1-1-0-0.yaml"

cfg = "experiments/final/newganpipeline_tex256_f16_data5-8_loss1-10-10-10.yaml"




def main():    
    update_config(cfg)
    logger, final_output_dir, tb_log_dir, checkpoint_dir = create_logger(cfg, 'valid')
    # cudnn setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("use device: %s", device)

    # define model
    netG = NewGanPipelineModel(configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1], configs.NEURAL_TEXTURE.FEATURE_NUM, device).to(device)

    # resume model from a checkpoint
    resume_file = join(checkpoint_dir, 'checkpoint.pth.tar')
    # resume_file = join(checkpoint_dir, 'model_best.pth.tar')
    if isfile(resume_file):
        logger.info("=> loading checkpoint '{}'".format(resume_file))
        try:
            checkpoint = torch.load(resume_file)
        except Exception:
            logger.info("=> map_location='cpu'")
            checkpoint = torch.load(resume_file, map_location='cpu')
        configs.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
        min_loss = checkpoint['min_loss']

        state_dict_old_G = checkpoint['state_dict_G']
        netG.load_state_dict(state_dict_old_G)

        logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(resume_file, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(resume_file))

    # create dataLoader
    val_dataset = EggDataset(root=configs.DATASET.ROOT, is_train=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    # valid data
    netG.eval()

    MSE_acc = AverageMeter()
    MAE_acc = AverageMeter()
    SSIM_acc = AverageMeter()

    with torch.no_grad():

        for i, (uv_map, gt, masks, normal, light_pos, view_dir) in enumerate(val_loader):
                
            gt = torch.log(math.exp(-3)+gt) / 3

            uv_map = uv_map.to(device)
            gt = gt.to(device)
            masks = masks.to(device)
            normal = normal.to(device)
            light_pos = light_pos.to(device)
            view_dir = view_dir.to(device)

            # compute output
            real_A, fake_B = netG(uv_map, normal, view_dir, light_pos)
            real_B = gt

            MAE, MSE, SSIM = accuracy(fake_B, real_B, masks)

            MAE_acc.update(MAE.item(), uv_map.size(0))
            MSE_acc.update(MSE.item(), uv_map.size(0))
            SSIM_acc.update(SSIM, uv_map.size(0))

            if i % 30 == 0 or i == len(val_loader) - 1:
                logger.info('[%d/%d] MAE: %.4f MSE: %.4f SSIM: %.4f' % (i, len(val_loader), MAE_acc.avg, MSE_acc.avg, SSIM_acc.avg))

main()