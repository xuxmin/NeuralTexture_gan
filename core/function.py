import time
import torch
import logging
import math

from core.config import configs
from core.utils.evaluation import AverageMeter
from core.utils.imutils import imwrite

logger = logging.getLogger(__name__)


def save_tensor_image(image, name):
    from core.utils.imutils import im_to_numpy
    import numpy as np
    import imageio

    npimg = im_to_numpy(image * 255).astype(np.uint8)
    imageio.imwrite(name, npimg)


def train(train_loader, model, criterion, optimizer, epoch, writer_dict, device):
    """
    train for one epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, (uv_map, gt, masks, normal, light_pos, view_dir) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if configs.TRAIN.PROCESS:
            gt = 0
        else:
            gt = torch.log(math.exp(-3)+gt) / 3

        uv_map = uv_map.to(device)
        gt = gt.to(device)
        normal = normal.to(device)
        light_pos = light_pos.to(device)
        view_dir = view_dir.to(device)

        if configs.MODEL.NAME == 'pipeline':
            _, output = model(uv_map, view_dir, light_pos)
        elif configs.MODEL.NAME == 'linear_pipeline':
            _, output = model(uv_map, view_dir, light_pos)
        elif configs.MODEL.NAME == 'newganpipeline':
            _, output = model(uv_map, normal, view_dir, light_pos)

        masks = (masks == 1).to(device)
        mask_preds = torch.masked_select(output, masks)
        mask_gt = torch.masked_select(gt, masks)
        loss = criterion(mask_preds, mask_gt)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.update(loss.item(), uv_map.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % configs.PRINT_FREQ == 0 or i == len(train_loader)-1:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=uv_map.size(0)/batch_time.val,
                    data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.avg, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            imgA = torch.exp(output[0].detach().cpu() * 3) - math.exp(-3)
            imgB = torch.exp(gt[0].detach().cpu() * 3) - math.exp(-3)
            save_tensor_image(imgA, "output/pred.png")
            save_tensor_image(imgB, "output/gt.png")

    return losses.avg


def validate(val_loader, model, criterion, epoch, writer_dict, device):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (uv_map, gt, masks, normal, light_pos, view_dir) in enumerate(val_loader):

            data_time.update(time.time() - end)

            if configs.TRAIN.PROCESS:
                gt = 0
            else:
                gt = torch.log(math.exp(-3)+gt) / 3

            uv_map = uv_map.to(device)
            gt = gt.to(device)
            normal = normal.to(device)
            light_pos = light_pos.to(device)
            view_dir = view_dir.to(device)

            # compute output
            _, output = model(uv_map, normal, view_dir, light_pos)


            masks = (masks == 1).to(device)
            mask_preds = torch.masked_select(output, masks)
            mask_gt = torch.masked_select(gt, masks)
            loss = criterion(mask_preds, mask_gt)

            # loss = criterion(output, gt)

            losses.update(loss.item(), uv_map.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.PRINT_FREQ == 0 or i == len(val_loader)-1:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses)
                logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return losses.avg