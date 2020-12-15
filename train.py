"""
NeuralTexture 原论文方法训练
"""

import argparse
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

from core.models.pipeline import PipelineModel
from core.models.pipeline import NewGanPipelineModel
from core.datasets.egg import EggDataset

from core.config import configs
from core.config import update_config
from core.utils.misc import create_logger
from core.utils.misc import save_checkpoint
from core.utils.osutils import join
from core.utils.osutils import isfile

from core.function import train, validate


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, _ = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=configs.PRINT_FREQ,
                        type=int)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir, checkpoint_dir = create_logger(args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(configs))

    min_loss = 10000

    # cudnn setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("use device: %s", device)

    # define model
    if configs.MODEL.NAME == 'pipeline':
        model = PipelineModel(configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1], configs.NEURAL_TEXTURE.FEATURE_NUM, device).to(device)
    elif configs.MODEL.NAME == 'linear_pipeline':
        model = LinearPipelineModel(configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1], configs.NEURAL_TEXTURE.FEATURE_NUM, device).to(device)
    else:
        model = NewGanPipelineModel(configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1], configs.NEURAL_TEXTURE.FEATURE_NUM, device).to(device)


    model.eval()
    # tensorboardX draw model
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # set loss function
    criterion = torch.nn.L1Loss(reduction='mean')

    # set optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.TRAIN.LR)


    # set lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, configs.TRAIN.LR_STEP, configs.TRAIN.LR_FACTOR
    )

    # resume model from a checkpoint
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

        state_dict_old = checkpoint['state_dict']
        model.load_state_dict(state_dict_old)

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(resume_file, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(resume_file))
        logger.info("=> create new checkpoint at '{}'".format(resume_file))

    # create dataLoader
    train_dataset = EggDataset(root=configs.DATASET.ROOT, is_train=True)
    val_dataset = EggDataset(root=configs.DATASET.ROOT, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs.TRAIN.BATCH_SIZE,
        shuffle=configs.TRAIN.SHUFFLE,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.TEST.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    # start training
    for epoch in range(configs.TRAIN.BEGIN_EPOCH, configs.TRAIN.END_EPOCH):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer_dict, device)

        # evaluate on validation set
        valid_loss = validate(val_loader, model, criterion, epoch, writer_dict, device)

        lr_scheduler.step()
        print (optimizer.param_groups[0]['lr'])

        # remember best acc and save checkpoint
        is_best = valid_loss < min_loss
        min_loss = min(valid_loss, min_loss)

        logger.info('=> saving checkpoint to {}'.format(checkpoint_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_dir)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()