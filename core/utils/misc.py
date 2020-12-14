
import os
import shutil
import time
import logging
import torch
from pathlib import Path
from core.utils.osutils import join
from core.config import configs


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def create_logger(cfg_name, phase='train'):
    """
    create the dir for the tensorboard log, model's output and the checkpoint
    according to the different datasets and models
    """
    root_output_dir = Path(configs.OUTPUT_DIR)

    # create output dir
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = configs.DATASET.NAME
    dataset = dataset.replace(':', '_')
    model = configs.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    # create specific dir
    final_output_dir = root_output_dir / dataset / model / cfg_name
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # setup logger
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = Path(configs.LOG_DIR) / dataset / model / cfg_name / log_file
    
    head = '%(asctime)-15s %(message)s'

    if final_log_file.exists():
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
    else:
        final_log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=str(final_log_file),
                            filemode='w',
                            format=head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(configs.LOG_DIR) / dataset / model / cfg_name
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(configs.LOG_DIR) / dataset / model / cfg_name
    print('=> creating {}'.format(checkpoint_dir))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir), str(checkpoint_dir)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    """
    Args:
    - state: all infomation of the model, such as, epoch, arch, state_dict, best_acc, optimizer
    - is_best: whether the model is best or not
    - checkpoint: the dir path to save checkpoint
    """
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth.tar'))
