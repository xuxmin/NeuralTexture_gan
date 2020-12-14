import logging
import torch
import cv2
import numpy as np
import math
from torch.utils.data import Dataset
import torch.nn.functional as F

from core.utils.camutils import load_bin
from core.utils.imutils import load_image
from core.utils.imutils import load_png
from core.utils.imutils import resize
from core.utils.imutils import augment
from core.utils.imutils import CEToneMapping
from core.utils.imutils import to_torch
from core.utils.imutils import im_to_torch
from core.utils.osutils import isfile
from core.config import configs

logger = logging.getLogger(__name__)


def sample_texture(texture, sample):
    """
    texture: 1 × 1 × H × W
    sample: 1 × H × W × 2

    Return: 3 × H × W
    """
    texture = texture.unsqueeze(0)
    texture_R = F.grid_sample(texture[:, 0:1, :, :], sample, align_corners=True)
    texture_G = F.grid_sample(texture[:, 1:2, :, :], sample, align_corners=True)
    texture_B = F.grid_sample(texture[:, 2:3, :, :], sample, align_corners=True)
    result = torch.cat(tuple([texture_R, texture_G, texture_B]), dim=1)
    return result[0]


class EggDataset(Dataset):
    def __init__(self, root, is_train=False):
        """
        root: "D:\\Code\\Project\\NeuralTexture\\data"
        """
        self.root = root
        self.is_train = is_train
        self.train_data = []
        self.valid_data = []
        self.light_data = load_bin("{}\\lights_8x8.bin".format(root), (2, 384, 3))  # pos, norm

        self.normal_map = load_image("{}\\normal_geo_gloabl.exr".format(root))

        self._split_dataset(configs.DATASET.MODE)


    def _load_normal_map(self):
        self.normal_map_list = []
        for i in range(32):
            img = load_image('{}\\gt\\{}\\result1024\\normal_geo_gloabl.exr'.format(self.root, i))
            img = (img - 0.5) * 2               # convert to range [-1, 1]
            self.normal_map_list.append(img)


    def _load_gutter_pos(self):
        self.gutter_pos_list = []
        for i in range(32):
            gutter_map = load_image('{}\\gt\\{}\\result1024_\\gutter_map.exr'.format(self.root, i), False)
            gutter_pos = np.argwhere(gutter_map[:, :, 0] > 0)
            self.gutter_pos_list.append(gutter_pos)


    def _split_dataset(self, mode):
        
        if mode == "TEST_LIGHT":
            for folder in range(32):
                for p in range(384):
                    if folder == 0 and p % 10 == 0:
                        self.valid_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
                    elif folder == 1:
                        self.train_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
        elif mode == "TEST_VIEW":
            for folder in range(32):
                for p in range(384):
                    if p == 0 and folder % 8 == 0:
                        self.valid_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
                    elif p == 0:
                        self.train_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
        elif mode == "TEST_ALL":
            for folder in range(32):
                for p in range(384):
                    if folder % 8 != 0 and folder != 1:
                        continue
                    if folder % 8 == 0 and p % 2 == 0:   # 取4个方向(0, 8, 16, 24), 每个方向192种光照, 共 192*4=768 张图片
                        self.train_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
                    elif p % 8 == 0:
                        self.valid_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
        elif mode == "TEST_MORE":
            for folder in range(32):
                for p in range(384):
                    if folder % 8 != 7 and p % 8 == 0:  # 取 28 个方向(0-6, 8-14, 16-22, 24-30), 每个方向 48 种光照, 共 28*48=1344 张图片
                        self.train_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
                    elif folder == 7 and p % 8 == 0:    # 新视角, 旧方向
                        self.valid_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
                    elif (folder == 7 or folder == 23)and p % 8 == 1:    # 新视角, 新方向
                        self.valid_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
                    elif folder == 8 and p % 8 == 1:    # 旧视角, 新方向
                        self.valid_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
        elif mode == 'ALL_DATA_7/8':
            for folder in range(32):
                for p in range(384):
                    if folder % 8 != 1 and p % 8 != 1:              # 取 28 个方向, 每个方向 336 种光照, 一共 9408 张图像
                        self.train_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
                    elif folder == 1 and p % 2 == 1:                # 随便取一点作为验证集
                        self.valid_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder, p))
        elif mode == 'ONE_VIEW':
            for p in range(64):
                self.train_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, 0, p))
                self.valid_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, 0, p))
        elif mode == 'TEST_6':
            for p in [0, 10, 20, 30, 40, 50]:
                self.train_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, 0, p))
                # self.valid_data.append("{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, 0, p))
        


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

    def getData(self, folder_idx, image_idx):

        path = "{}\\gt\\{}\\img{:0>5d}_cam00.exr".format(self.root, folder_idx, image_idx)

        if configs.DATASET.DLV:
            return self._getitem_dlv(path, False)
        else:
            return self._getitem(path, False)

    def getLightDirMap(self, folder_idx, image_idx, light_position):
        
        gutter_pos_path = '{}\\gt\\{}\\result1024_\\gutter_pos.pt'.format(self.root, folder_idx)

        if isfile(gutter_pos_path):
            gutter_pos = torch.load(gutter_pos_path)
        else:
            gutter_map = load_image('{}\\gt\\{}\\result1024_\\gutter_map.exr'.format(self.root, folder_idx), False)
            gutter_pos = np.argwhere(gutter_map[:, :, 0] > 0)
            torch.save(gutter_pos, gutter_pos_path)
    
        # position_origin: in result1024_ !!!
        position_origin = load_bin("{}\\gt\\{}\\result1024_\\positions_origin.bin".format(self.root, folder_idx), (gutter_pos.shape[0], 3))

        light_dir = light_position - position_origin
        light_dir = light_dir / np.sqrt(np.sum(light_dir * light_dir, axis=1, keepdims=True))
        light_dir_map = -np.ones((1024, 1024, 3))
        for i, p in enumerate(gutter_pos):
            light_dir_map[p[0], p[1]] = light_dir[i]
        light_dir_map = im_to_torch(light_dir_map)

        return light_dir_map
    
    def getViewDirMap(self, folder_idx, image_idx, camera_position):

        gutter_pos_path = '{}\\gt\\{}\\result1024_\\gutter_pos.pt'.format(self.root, folder_idx)

        if isfile(gutter_pos_path):
            gutter_pos = torch.load(gutter_pos_path)
        else:
            gutter_map = load_image('{}\\gt\\{}\\result1024_\\gutter_map.exr'.format(self.root, folder_idx), False)
            gutter_pos = np.argwhere(gutter_map[:, :, 0] > 0)
            torch.save(gutter_pos, gutter_pos_path)

        # position_origin: in result1024_ !!!
        position_origin = load_bin("{}\\gt\\{}\\result1024_\\positions_origin.bin".format(self.root, folder_idx), (gutter_pos.shape[0], 3))

        view_dir = camera_position - position_origin
        view_dir = view_dir / np.sqrt(np.sum(view_dir * view_dir, axis=1, keepdims=True))
        view_dir_map = -np.ones((1024, 1024, 3))
        for i, p in enumerate(gutter_pos):
            view_dir_map[p[0], p[1]] = view_dir[i]
        view_dir_map = im_to_torch(view_dir_map)

        return view_dir_map

    def _getitem_dlv(self, path, do_augment):
        """
        dlv: different light dir/ view dir

        假设物体不动, 相机和光源在旋转
        
        Returns:
        - uv_map:
        - gt: ground truth 经过裁剪, 范围仍是 [0, 正无穷大]
        - mask:
        - normal_map:
        - light_map:
        - view_map:  
        """
        folder_idx, image_idx = self._parse_path(path)

        uv_map_path =  self.root + "\\gt\\{}\\result1024\\render_extrinsic.yml.exr".format(folder_idx)
        mask_path =  self.root + "\\gt\\{}\\mask_cam00.png".format(folder_idx)

        # load uv_map
        uv_map = load_image(uv_map_path)        # C × H × W,  [0, 1]
        uv_map = uv_map[:, 275:625, 325:675]    # clip

        # load mask
        mask_path_tmp = mask_path.replace('.png', '.pt')
        if isfile(mask_path_tmp):
            mask = torch.load(mask_path_tmp)
        else:
            mask = load_png(mask_path)
            torch.save(mask, mask_path_tmp)
        mask = mask[:, 1100:2500, 1300:2700]    # clip 
        
        # load gt
        gt_path = path.replace('.exr', '.npy')
        gt = to_torch(np.load(gt_path))
        gt[mask == 0] = 0                       # mask gt

        # resize uv_map, mask, gt to [350, 350]
        gt = resize(gt, 350, 350)
        uv_map = resize(uv_map, 350, 350)
        mask = resize(mask, 350, 350)

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

        # light_position
        light_position = self.light_data[0][image_idx]
        # camera_position
        camera_position = np.array([353.068, 0.833654, 316.206])

        # rotate light and camera
        rot_angle = folder_idx * 3.14159 / 2 / 8
        transform_matrix = np.array(
            [
                [ math.cos(rot_angle), math.sin(rot_angle), 0],
                [-math.sin(rot_angle), math.cos(rot_angle), 0],
                [ 0, 0, 1]
            ]
        )
        light_position = np.matmul(transform_matrix, light_position)
        camera_position = np.matmul(transform_matrix, camera_position)

        light_dir_map = self.getLightDirMap(folder_idx, image_idx, light_position)
        view_dir_map = self.getViewDirMap(folder_idx, image_idx, camera_position)

        view_dir = sample_texture(view_dir_map, sample)
        light_dir = sample_texture(light_dir_map, sample)


        return uv_map, gt, mask, normal, light_dir, view_dir

    def _getitem(self, path, do_augment):
        """
        下面是假设物体不动, 相机和光源绕着物体旋转, 坐标系统一为 竖轴为z, 横轴为 y, 朝外的为 x
        - 法向量是一致的, 不需要旋转
        - view_dir 需要旋转
        - light_pos 需要旋转
        """

        folder_idx, image_idx = self._parse_path(path)

        uv_map_path =  self.root + "\\gt\\{}\\result1024\\render_extrinsic.yml.exr".format(folder_idx)
        mask_path =  self.root + "\\gt\\{}\\mask_cam00.png".format(folder_idx)

        # load uv_map
        uv_map = load_image(uv_map_path)        # C × H × W,  [0, 1]
        uv_map = uv_map[:, 275:625, 325:675]

        # load mask
        mask_path_tmp = mask_path.replace('.png', '.pt')
        if isfile(mask_path_tmp):
            mask = torch.load(mask_path_tmp)
        else:
            mask = load_png(mask_path)
            torch.save(mask, mask_path_tmp)
        mask = mask[:, 1100:2500, 1300:2700]

        # load gt
        tmp = path.replace('.exr', '.npy')
        gt = to_torch(np.load(tmp))
        gt[mask == 0] = 0

        # resize
        gt = resize(gt, 350, 350)
        uv_map = resize(uv_map, 350, 350)
        mask = resize(mask, 350, 350)

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

        # image_idx = self.calculate_id(image_idx)
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

    def __getitem__(self, index):

        if self.is_train:
            path = self.train_data[index]
        else:
            path = self.valid_data[index]

        if self.is_train and configs.DATASET.AUGMENT:
            if configs.DATASET.DLV:
                return self._getitem_dlv(path, True)
            else:
                return self._getitem(path, True)
        else:
            if configs.DATASET.DLV:
                return self._getitem_dlv(path, False)
            else:
                return self._getitem(path, False)


"""
    def _getitem_dlv(self, path, do_augment):
        # dlv: different light dir/ view dir

        # 下面假设相机与光源位置始终不变, 物体在旋转
        
        # Returns:
        # - uv_map:
        # - gt: ground truth 经过裁剪, 范围仍是 [0, 正无穷大]
        # - mask:
        # - normal_map: 随物体旋转而改变, range [-1, 1]
        # - light_map:
        # - view_map:  
        folder_idx, image_idx = self._parse_path(path)

        uv_map_path =  self.root + "\\gt\\{}\\result1024\\render_extrinsic.yml.exr".format(folder_idx)
        mask_path =  self.root + "\\gt\\{}\\mask_cam00.png".format(folder_idx)

        # load uv_map
        uv_map = load_image(uv_map_path)        # C × H × W,  [0, 1]
        uv_map = uv_map[:, 275:625, 325:675]    # clip

        # load mask
        mask_path_tmp = mask_path.replace('.png', '.pt')
        if isfile(mask_path_tmp):
            mask = torch.load(mask_path_tmp)
        else:
            mask = load_png(mask_path)
            torch.save(mask, mask_path_tmp)
        mask = mask[:, 1100:2500, 1300:2700]    # clip 
        
        # load gt
        gt_path = path.replace('.exr', '.npy')
        gt = to_torch(np.load(gt_path))
        gt[mask == 0] = 0                       # mask gt

        # -----------------------------------------------
        # gutter_pos
        gutter_pos = self.gutter_pos_list[folder_idx]
        # position_origin
        position_origin = load_bin("{}\\gt\\{}\\result1024\\positions_origin.bin".format(self.root, folder_idx), (gutter_pos.shape[0], 3))
        # normal_geo
        normals_geo = load_bin("{}\\gt\\{}\\result1024\\normals_geo.bin".format(self.root, folder_idx), (gutter_pos.shape[0], 3))

        # light_position
        light_position = self.light_data[0][image_idx]
        # camera_position
        camera_position = np.array([353.068, 0.833654, 316.206])

        light_dir = light_position - position_origin
        view_dir = camera_position - position_origin
        # normalize
        light_dir = light_dir / np.sqrt(np.sum(light_dir * light_dir, axis=1, keepdims=True))
        view_dir = view_dir / np.sqrt(np.sum(view_dir * view_dir, axis=1, keepdims=True))

        # light_dir_map, view_dir_map, normal_map
        light_dir_map = -np.ones((1024, 1024, 3))
        view_dir_map = -np.ones((1024, 1024, 3))
        normal_map = -np.ones((1024, 1024, 3))
        for i, p in enumerate(gutter_pos):
            light_dir_map[p[0], p[1]] = light_dir[i]
            view_dir_map[p[0], p[1]] = view_dir[i]
            normal_map[p[0], p[1]] = normals_geo[i]
        light_dir_map = im_to_torch(light_dir_map)
        view_dir_map = im_to_torch(view_dir_map)
        normal_map = im_to_torch(normal_map)
        # ------------------------------------------------

        # resize uv_map, mask, gt to [350, 350]
        gt = resize(gt, 350, 350)
        uv_map = resize(uv_map, 350, 350)
        mask = resize(mask, 350, 350)

        # data argument when training the model
        if do_augment:
            uv_map, gt, mask = augment([uv_map, gt, mask], 350)

        uv_map = resize(uv_map, configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1])
        gt = resize(gt, configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1])
        mask = resize(mask, configs.MODEL.IMAGE_SIZE[0], configs.MODEL.IMAGE_SIZE[1])

        sample = uv_map.permute(1, 2, 0)[:, :, :2].unsqueeze(0)            # 1 × H × W × 2
        sample = 2 * sample - 1
        sample[:, :, :, 1] = -sample[:, :, :, 1]

        normal = sample_texture(normal_map, sample)
        view_dir = sample_texture(view_dir_map, sample)
        light_dir = sample_texture(light_dir_map, sample)

        return uv_map, gt, mask, normal, light_dir, view_dir
"""