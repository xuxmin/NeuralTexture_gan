import logging
import torch
import cv2
import numpy as np
import math
from torch.utils.data import Dataset
import torch.nn.functional as F
import os

from core.utils.camutils import load_bin
from core.utils.imutils import load_image
from core.utils.imutils import load_png
from core.utils.imutils import resize
from core.utils.imutils import augment
from core.utils.imutils import to_torch
from core.utils.imutils import im_to_torch
from core.utils.imutils import im_to_numpy
from core.utils.osutils import isfile
from core.config import configs

logger = logging.getLogger(__name__)


class BottleDataset(Dataset):
    def __init__(self, root, is_train=False):
        self.root = root
        self.is_train = is_train
        self.train_data = []
        self.valid_data = []
        self.light_data = load_bin("{}\\lights_8x8.bin".format(root), (2, 384, 3))  # pos, norm
        