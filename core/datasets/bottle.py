import logging
import torch
import cv2
import numpy as np
import math
from torch.functional import norm
from torch.utils.data import Dataset
import os

from core.utils.camutils import load_bin
from core.utils.imutils import load_image
from core.utils.imutils import load_png
from core.utils.imutils import resize
from core.utils.imutils import augment
from core.utils.imutils import to_torch
from core.utils.imutils import im_to_torch
from core.utils.imutils import im_to_numpy
from core.utils.imutils import sample_texture
from core.utils.osutils import isfile, join
from core.config import configs

logger = logging.getLogger(__name__)


def get_uv_map(camera_pos):
    """
    camera_pos: (3, )
    """

    oldval = os.getcwd()                                    # 查看当前工作目录
    os.chdir('D:\\Code\\Project\\tools\\PathTracer_NEWCAM')   # 修改当前工作目录

    x = camera_pos[0]
    y = camera_pos[1]
    z = camera_pos[2]

    file_name = "render-{}-{}-{}".format(x, y, z)
    file_path = "res\\bottle_bak\\{}.exr".format(file_name)

    if not isfile(file_path):
        os.system('.\\PathTracer_NEWCAM_UV.exe obj\\test\\bottle_bak_crop.obj uv {} {} {} {}'.format(x, y, z, file_path))

    img = load_image(file_path)

    os.chdir(oldval)

    if img is None:
        print("error get_uv_map")

    return img


def get_uv_map_rot(camera_pos, rotation):
    oldval = os.getcwd()                                    # 查看当前工作目录
    os.chdir('D:\\Code\\Project\\tools\\PathTracer_NEWCAM_ROT')   # 修改当前工作目录

    x = camera_pos[0]
    y = camera_pos[1]
    z = camera_pos[2]

    file_name = "render-{}-{}-{}-{}".format(x, y, z, rotation)
    file_path = "res\\bottle_bak\\{}.exr".format(file_name)

    if not isfile(file_path):
        os.system('.\\PathTracer_ROT_TMP.exe obj\\test\\bottle_bak_crop.obj uv {} {} {} {} {}'.format(x, y, z, file_path, rotation))

    img = load_image(file_path)

    os.chdir(oldval)

    if img is None:
        print("error get_uv_map")

    return img


class BottleDataset(Dataset):

    def __init__(self, root, is_train=False):
        self.root = root
        self.is_train = is_train
        self.train_data = []
        self.valid_data = []
        self.light_data = load_bin(join(self.root, "lights_8x8.bin"), (2, 384, 3))  # pos, norm

        normal_path = join(self.root, "normal_geo_gloabl.pt")

        if isfile(normal_path):
            self.normal_map = torch.load(normal_path)
        else:
            self.normal_map = load_image(join(self.root, "normal_geo_gloabl.exr"))
            torch.save(self.normal_map, normal_path)
        
        self._split_dataset(configs.DATASET.MODE)
    
    def _split_dataset(self, mode):

        if mode == "ALL_DATA":
            for folder in range(configs.DATASET.VIEW_NUM):
                for p in range(configs.DATASET.LIGHT_NUM):
                    path = join(self.root, "gt", str(folder), "img{:0>5d}_cam00.exr".format(p))

                    if not isfile(path):
                        logger.warning("image {} is not existed!".format(path))
                        continue

                    self.train_data.append(path)
                    if folder == 10 and p % 8 == 0:
                        self.valid_data.append(path)
        else:
            exit()

    def _parse_path(self, path):
        """
        'D:/Code/Project/NeuralTexture/data/gt/13/img00073_cam00.exr'
        return (13, 73)
        """
        path = path.replace('/', '\\')
        path = path.split('\\')
        folder = int(path[-2])
        image_name = path[-1]
        image_idx = int(image_name.split('_')[0][-5:])
        return folder, image_idx

    def __len__(self):
        if self.is_train:
            return len(self.train_data)
        else:
            return len(self.valid_data)
    
    def __getitem__(self, index):
        if self.is_train:
            path = self.train_data[index]
        else:
            path = self.valid_data[index]

        if self.is_train and configs.DATASET.AUGMENT:
            return self._getitem(path, True)
        else:
            return self._getitem(path, False)
    
    def _getitem(self, path, do_augment):
        """
        下面是假设物体不动, 相机和光源绕着物体旋转, 坐标系统一为 竖轴为z, 横轴为 y, 朝外的为 x
        - 法向量是一致的, 不需要旋转
        - view_dir 需要旋转
        - light_pos 需要旋转
        """
        folder_idx, image_idx = self._parse_path(path)

        uv_map_path = join(self.root, "gt", str(folder_idx), "result256", "crop_render_extrinsic.yml.exr")
        mask_path = join(self.root, "gt", str(folder_idx), "mask_cam00.png")

        uv_map_pt_path = join(self.root, "gt", str(folder_idx), "result256", "crop_render_extrinsic.yml.pt")
        if isfile(uv_map_pt_path):
            uv_map = torch.load(uv_map_pt_path)
        else:
            uv_map = load_image(uv_map_path)        # C × H × W,  [0, 1]
            torch.save(uv_map, uv_map_pt_path)

        # load uv_map
        # -------------
        # 需要合适的裁剪方法
        # -------------
        uv_map = uv_map[:, 300:900, 350:950]

        # load mask
        mask_path_tmp = mask_path.replace('.png', '.pt')
        if isfile(mask_path_tmp):
            mask = torch.load(mask_path_tmp)
        else:
            mask = load_png(mask_path)
            torch.save(mask, mask_path_tmp)
        # -----------
        # 需要合适的裁剪方法与 uv_map 对上
        # -----------
        mask = mask[:, 900:3600, 1400:3800]

        # load gt
        tmp = path.replace('.exr', '.pt')
        if isfile(tmp):
            gt = torch.load(tmp)
        else:
            gt = load_image(path)
            gt = gt[:, 900:3600, 1400:3800]
            gt = resize(gt, 350, 350)
            torch.save(gt, tmp)

        # resize
        gt = resize(gt, 350, 350)
        uv_map = resize(uv_map, 350, 350)
        mask = resize(mask, 350, 350)

        # do mask
        gt[mask == 0] = 0

        # data argument when training the model
        if do_augment:
            uv_map, gt, mask = augment([uv_map, gt, mask], 350)

        uv_map = resize(uv_map, configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1])
        gt = resize(gt, configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1])
        mask = resize(mask, configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1])

        sample = uv_map.permute(1, 2, 0)[:, :, :2].unsqueeze(0)            # 1 × H × W × 2
        sample = 2 * sample - 1
        sample[:, :, :, 1] = -sample[:, :, :, 1]

        normal = sample_texture((self.normal_map - 0.5)*2, sample)

        rot_angle = folder_idx * 3.14159 / 2 / 8
        view_dir = np.zeros((3, ))
        view_dir[0] = math.cos(-rot_angle)
        view_dir[1] = math.sin(-rot_angle)
        view_dir[2] = 0

        light_pos = self.light_data[0][image_idx]

        transform_matrix = np.array(
            [
                [ math.cos(rot_angle), math.sin(rot_angle), 0],
                [-math.sin(rot_angle), math.cos(rot_angle), 0],
                [ 0, 0, 1]
            ]
        )

        light_pos = np.matmul(transform_matrix, light_pos)

        light_pos = light_pos / math.sqrt(np.sum(light_pos * light_pos))

        light_pos = torch.from_numpy(light_pos)
        view_dir = torch.from_numpy(view_dir)

        return uv_map, gt, mask, normal, light_pos, view_dir

    def getData2(self, camera_pos, rotation=None):
        if rotation:
            uv_map = get_uv_map_rot(camera_pos, rotation)
        else:
            uv_map = get_uv_map(camera_pos)

        # 获取对应的 ROI
        # roi = ThresholdROI(im_to_numpy(uv_map), thresh=0.0001)
        # x, y, w, h = roi

        x, y, w, h = 400, 400, 500, 500

        # 根据 roi 裁剪出 uv_map
        uv_map = uv_map[:, x:x+w, y:y+h]

        size = w if w > h else h
        # size = size + 50

        # 填充边缘弄成一个正方形
        np_uv_map = im_to_numpy(uv_map)
        pad_w = (size - w) // 2
        pad_h = (size - h) // 2

        pad_img = cv2.copyMakeBorder(np_uv_map, pad_w, pad_w, pad_h, pad_h, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        pad_img = im_to_torch(pad_img)

        # 缩放到目标大小
        uv_map = resize(pad_img, configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1])

        sample = uv_map.permute(1, 2, 0)[:, :, :2].unsqueeze(0)            # 1 × H × W × 2
        sample = 2 * sample - 1
        sample[:, :, :, 1] = -sample[:, :, :, 1]

        normal = sample_texture((self.normal_map - 0.5)*2, sample)

        return uv_map, normal