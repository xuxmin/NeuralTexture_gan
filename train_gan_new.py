import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import cv2
import math
import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse

from core.models.pipeline import NewGanPipelineModel
from core.models.discriminator import Discriminator
from core.models.vggnet import VGG19
from core.datasets.base_dataset import BaseDataset

from core.config import configs
from core.config import update_config
from core.utils.misc import create_logger
from core.utils.misc import save_checkpoint
from core.utils.osutils import join
from core.utils.osutils import isfile
from core.utils.osutils import exists
from core.utils.osutils import dirname
from core.utils.osutils import mkdir_p
from core.utils.imutils import show_img_list
from core.utils.imutils import CEToneMapping
from core.utils.imutils import ACESToneMapping
from core.utils.imutils import im_to_numpy
from core.utils.evaluation import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str,
                        )
    args, _ = parser.parse_known_args()

    update_config(args.cfg)

    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=configs.PRINT_FREQ,
                        type=int)
    args = parser.parse_args()

    return args


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def save_tensor_image(image, name):
    from core.utils.imutils import im_to_numpy
    import numpy as np
    import imageio

    # tone_img = CEToneMapping(image, 3)
    tone_img = ACESToneMapping(image, 10)

    npimg = im_to_numpy(tone_img * 255).astype(np.uint8)
    imageio.imwrite(name, npimg)


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir, checkpoint_dir = create_logger(args.cfg, 'train')

    # cudnn setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("use device: %s", device)

    # define model
    netG = NewGanPipelineModel(configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1], configs.NEURAL_TEXTURE.FEATURE_NUM, device).to(device)
    netD = Discriminator(configs.NEURAL_TEXTURE.FEATURE_NUM, 3, 64).to(device)
    # netD = Discriminator2(configs.NEURAL_TEXTURE.FEATURE_NUM, 3, 64).to(device)
    vggNet = VGG19().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # tensorboardX draw model
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # set loss function
    criterion = nn.BCELoss()
    criterionL1 = nn.L1Loss()

    # set optimizer
    optimizerD = torch.optim.SGD(netD.parameters(), lr=configs.TRAIN.D_LR, momentum=0.8)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=configs.TRAIN.G_LR)

    # set lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizerG, configs.TRAIN.LR_STEP, configs.TRAIN.LR_FACTOR
    )

    min_loss = 10000

    resume_file = join(checkpoint_dir, 'checkpoint.pth.tar')
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
        state_dict_old_D = checkpoint['state_dict_D']
        netD.load_state_dict(state_dict_old_D)
        netG.load_state_dict(state_dict_old_G)

        optimizerD.load_state_dict(checkpoint['optimizer_D'])
        optimizerG.load_state_dict(checkpoint['optimizer_G'])
        
        if 'train_global_steps' in checkpoint:
            writer_dict['train_global_steps'] = checkpoint['train_global_steps']
            writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']

        logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(resume_file, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(resume_file))
        logger.info("=> create new checkpoint at '{}'".format(resume_file))
    
    # create dataLoader
    train_dataset = BaseDataset(root=configs.DATASET.ROOT, is_train=True, check_file_exist=True)
    val_dataset = BaseDataset(root=configs.DATASET.ROOT, is_train=False, check_file_exist=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs.TRAIN.BATCH_SIZE,
        shuffle=configs.TRAIN.SHUFFLE,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    # start training
    for epoch in range(configs.TRAIN.BEGIN_EPOCH, configs.TRAIN.END_EPOCH):

        netD.train()
        netG.train()

        loss_errD = AverageMeter()
        loss_errGAN = AverageMeter()
        loss_errL1 = AverageMeter()
        loss_errFM = AverageMeter()
        loss_errVGG = AverageMeter()

        for i, (uv_map, gt, masks, normal, light_pos, view_dir) in enumerate(train_loader):
            
            # -------------
            # 根据实际情况处理
            # -------------
            gt = torch.log(math.exp(-3)+gt) / 3

            uv_map = uv_map.to(device)
            gt = gt.to(device)
            normal = normal.to(device)
            light_pos = light_pos.to(device)
            view_dir = view_dir.to(device)

            # train D with real data
            optimizerD.zero_grad()

            real_A, fake_B = netG(uv_map, normal, view_dir, light_pos)
            real_B = gt

            real_AB = torch.cat((real_A, real_B), 1)
            _, output = netD(torch.autograd.Variable(real_AB))
            errD_real = criterion(output, torch.ones(output.size()).cuda())

            # train D with fake data
            fake_AB = torch.cat((real_A, fake_B), 1)
            _, output = netD(torch.autograd.Variable(fake_AB))
            errD_fake = criterion(output, torch.zeros(output.size()).cuda())

            errD = (errD_fake + errD_real) / 2
            errD.backward()
            optimizerD.step()

            # train G
            optimizerG.zero_grad()

            # GAN loss
            fake_features, output = netD(fake_AB)
            real_features, _ = netD(real_AB)
            errGAN = criterion(output, torch.ones(output.size()).cuda())
            # L1 loss
            masks = (masks == 1).to(device)
            mask_fake_B = torch.masked_select(fake_B, masks)
            mask_real_B = torch.masked_select(real_B, masks)
            errL1 = criterionL1(mask_fake_B, mask_real_B)
            # feature match loss
            errFM = 0
            for j in range(len(real_features)):
              errFM += criterionL1(real_features[j], fake_features[j])
            # vgg loss
            errVGG = 0
            weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            real_features_VGG, fake_features_VGG = vggNet(gt), vggNet(fake_B)
            for j in range(len(real_features_VGG)):
                errVGG += weights[j] * criterionL1(fake_features_VGG[j], real_features_VGG[j])

            errG = configs.TRAIN.LAMBDA_GAN * errGAN + configs.TRAIN.LAMBDA_L1 * errL1 + configs.TRAIN.LAMBDA_FM * errFM + configs.TRAIN.LAMBDA_VGG * errVGG

            errG.backward()
            optimizerG.step()


            loss_errD.update(errD.item(), uv_map.size(0))
            loss_errGAN.update(errGAN.item(), uv_map.size(0))
            loss_errL1.update(errL1.item(), uv_map.size(0))
            loss_errFM.update(errFM.item(), uv_map.size(0))
            loss_errVGG.update(errVGG.item(), uv_map.size(0))


            if i % 20 == 0:
                logger.info('[%d][%d/%d] Loss_D: %.4f(%.4f) Loss_G: %.4f(%.4f) Loss_L1: %.4f(%.4f) Loss_FM: %.4f(%.4f) Loss_VGG: %.4f(%.4f)'
                            % (epoch, i, len(train_loader), errD.item(), loss_errD.avg, errGAN.item(), loss_errGAN.avg, errL1.item(), loss_errL1.avg, errFM.item(), loss_errFM.avg, errVGG.item(), loss_errVGG.avg))
                
                imgA = torch.exp(fake_B[0].detach().cpu() * 3) - math.exp(-3)
                imgB = torch.exp(real_B[0].detach().cpu() * 3) - math.exp(-3)
                save_tensor_image(imgA, "output/pred.png")
                save_tensor_image(imgB, "output/gt.png")
        
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('loss_errD', errD.item(), global_steps)
                writer.add_scalar('loss_errGAN', errGAN.item(), global_steps)
                writer.add_scalar('loss_errL1', errL1.item(), global_steps)
                writer.add_scalar('loss_errFM', errFM.item(), global_steps)
                writer.add_scalar('loss_errVGG', errVGG.item(), global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        # valid data
        netG.eval()
        losses = AverageMeter()
        with torch.no_grad():
            logger.info ("Valid data:")
            for i, (uv_map, gt, masks, normal, light_pos, view_dir) in enumerate(val_loader):
                
                gt = torch.log(math.exp(-3)+gt) / 3

                uv_map = uv_map.to(device)
                gt = gt.to(device)
                normal = normal.to(device)
                light_pos = light_pos.to(device)
                view_dir = view_dir.to(device)

                # compute output
                real_A, fake_B = netG(uv_map, normal, view_dir, light_pos)
                real_B = gt

                # L1 loss
                masks = (masks == 1).to(device)
                mask_fake_B = torch.masked_select(fake_B, masks)
                mask_real_B = torch.masked_select(real_B, masks)
                errL1 = criterionL1(mask_fake_B, mask_real_B)

                losses.update(errL1.item(), uv_map.size(0))

                if i % 30 == 0:
                    logger.info('[%d][%d/%d] Loss_L1: %.4f(%.4f)' % (epoch, i, len(val_loader), errL1.item(), losses.avg))
                    imgA = torch.exp(fake_B[0].detach().cpu() * 3) - math.exp(-3)
                    imgB = torch.exp(real_B[0].detach().cpu() * 3) - math.exp(-3)
                    save_tensor_image(imgA, "output/valid_pred.png")
                    save_tensor_image(imgB, "output/valid_gt.png")
            
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

        lr_scheduler.step()

        valid_loss = losses.avg
        is_best = valid_loss < min_loss
        min_loss = min(valid_loss, min_loss)
                
        # remember best acc and save checkpoint
        logger.info('=> saving checkpoint to {}'.format(checkpoint_dir))

        name = 'checkpoint_epoch{}.pth.tar'.format(epoch)

        if epoch in [5, 10, 15, 20, 25]:
            save_checkpoint({
                'epoch': epoch + 1,
                'min_loss': min_loss,
                'train_global_steps': writer_dict['train_global_steps'],
                'valid_global_steps': writer_dict['valid_global_steps'],
                'state_dict_G': netG.state_dict(),
                'state_dict_D': netD.state_dict(),
                'optimizer_G': optimizerG.state_dict(),
                'optimizer_D': optimizerD.state_dict(),
            }, is_best, checkpoint=checkpoint_dir, filename=name)

        save_checkpoint({
                'epoch': epoch + 1,
                'min_loss': min_loss,
                'train_global_steps': writer_dict['train_global_steps'],
                'valid_global_steps': writer_dict['valid_global_steps'],
                'state_dict_G': netG.state_dict(),
                'state_dict_D': netD.state_dict(),
                'optimizer_G': optimizerG.state_dict(),
                'optimizer_D': optimizerD.state_dict(),
            }, is_best, checkpoint=checkpoint_dir)

    writer_dict['writer'].close()


if __name__ == "__main__":
    main()