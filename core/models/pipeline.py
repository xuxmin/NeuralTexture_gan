import torch
import torch.nn as nn
import math
import numpy as np
import logging

from .texture import Texture
from .generator import Generator
from .transformer_net import TransformerNet
from core.config import configs


logger = logging.getLogger(__name__)


class NewGanPipelineModel(nn.Module):

    def __init__(self, width, height, feature_num, device):
        super(NewGanPipelineModel, self).__init__()
        self.feature_num = feature_num
        self.width = width
        self.height = height
        self.texture = Texture(configs.NEURAL_TEXTURE.SIZE, configs.NEURAL_TEXTURE.SIZE, feature_num)

        if configs.MODEL.COMBINE == 'multiplicate':
            generator_channel = feature_num
        elif configs.MODEL.COMBINE == 'concatenate':
            generator_channel = feature_num + 9

        if configs.MODEL.GENERATE == 'Pix2Pix':
            self.generator = Generator(generator_channel, 3, 64)
        elif configs.MODEL.GENERATE == 'TransformerNet':
            self.generator = TransformerNet(generator_channel)

        self.device = device

    def forward(self, uv_map, normal, view_dir, light_dir):
        """
        Args:
        - uv_map: B × 3 × H × W
        - mask: B × 3 × H × W
        - normal: B × 3 × H × W
        - view_dir: B × 3 or B × 3 × H × W
        - light_dir: B × 3 or B × 3 × H × W
        """
        uv_map = uv_map.permute(0, 2, 3, 1)[:, :, :, :2]            # B × H × W × 2

        x = self.texture(uv_map)                                    # B × C × H × W

        if configs.MODEL.COMBINE == 'multiplicate':

            x[:, 3:6, :, :] = x[:, 3:6, :, :] * normal

            if not configs.DATASET.DLV:
                view_dir = view_dir.view(view_dir.shape[0], view_dir.shape[1], 1, 1)        # B × 3 × 1 × 1
                light_dir = light_dir.view(light_dir.shape[0], light_dir.shape[1], 1, 1)    # B × 3 × 1 × 1

            x[:, 6:9, :, :] = x[:, 6:9, :, :] * view_dir
            x[:, 9:12, :, :] = x[:, 9:12, :, :] * light_dir
        
        elif configs.MODEL.COMBINE == 'concatenate':

            if not configs.DATASET.DLV:
                view_dir = view_dir.view(view_dir.shape[0], view_dir.shape[1], 1, 1)     # B × 3 × H × W
                view_dir = view_dir.repeat(1, 1, self.height, self.width)
                view_dir = view_dir.type(torch.cuda.FloatTensor)

                light_dir = light_dir.view(light_dir.shape[0], light_dir.shape[1], 1, 1) # B × 3 × H × W
                light_dir = light_dir.repeat(1, 1, self.height, self.width)
                light_dir = light_dir.type(torch.cuda.FloatTensor)

            x = torch.cat(tuple([x, normal, view_dir, light_dir]), dim=1)

        else:
            x = 0

        y = self.generator(x)

        return x, y


class GanPipelineModel(nn.Module):

    def __init__(self, width, height, feature_num, device):
        super(GanPipelineModel, self).__init__()
        self.feature_num = feature_num
        self.width = width
        self.height = height
        self.texture = Texture(configs.NEURAL_TEXTURE.SIZE, configs.NEURAL_TEXTURE.SIZE, feature_num)
        self.generator = Generator(16, 3, 64)
        self.device = device

    def _spherical_harmonics_basis(self, extrinsics):
        '''
        Args:
        - extrinsics: a tensor shaped (B, 3)
        Returns: 
        - a tensor shaped (B, 9)
        '''
        batch = extrinsics.shape[0]
        sh_bands = torch.ones((batch, 9), dtype=torch.float)
        coff_0 = 1 / (2.0*math.sqrt(np.pi))
        coff_1 = math.sqrt(3.0) * coff_0
        coff_2 = math.sqrt(15.0) * coff_0
        coff_3 = math.sqrt(1.25) * coff_0
        # l=0
        sh_bands[:, 0] = coff_0
        # l=1
        sh_bands[:, 1] = extrinsics[:, 1] * coff_1
        sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        sh_bands[:, 3] = extrinsics[:, 0] * coff_1
        # l=2
        sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2
        sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2
        sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2
        sh_bands[:, 8] = (extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 2] * extrinsics[:, 2]) * coff_2
        return sh_bands

    def forward(self, uv_map, view_dir, light_dir):
        """
        Args:
        - uv_map: B × 3 × H × W
        - mask: B × 3 × H × W
        - view_dir: the position of the camera
        - light_dir: 
        """
        uv_map = uv_map.permute(0, 2, 3, 1)[:, :, :, :2]            # B × H × W × 2

        x = self.texture(uv_map)                                    # B × C × H × W

        basis = self._spherical_harmonics_basis(light_dir)
        basis = basis.to(self.device)
        basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)    # B × 9 × 1 × 1
        x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis
        
        y = self.generator(x)

        return x, y


class PipelineModel(nn.Module):

    def __init__(self, width, height, feature_num, device):
        super(PipelineModel, self).__init__()
        self.feature_num = feature_num
        self.width = width
        self.height = height
        self.texture = Texture(configs.NEURAL_TEXTURE.SIZE, configs.NEURAL_TEXTURE.SIZE, feature_num)
        self.device = device
        self.generator = Generator(16, 3, 64)

    def _spherical_harmonics_basis(self, extrinsics):
        '''
        Args:
        - extrinsics: a tensor shaped (B, 3)
        Returns: 
        - a tensor shaped (B, 9)
        '''
        batch = extrinsics.shape[0]
        sh_bands = torch.ones((batch, 9), dtype=torch.float)
        coff_0 = 1 / (2.0*math.sqrt(np.pi))
        coff_1 = math.sqrt(3.0) * coff_0
        coff_2 = math.sqrt(15.0) * coff_0
        coff_3 = math.sqrt(1.25) * coff_0
        # l=0
        sh_bands[:, 0] = coff_0
        # l=1
        sh_bands[:, 1] = extrinsics[:, 1] * coff_1
        sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        sh_bands[:, 3] = extrinsics[:, 0] * coff_1
        # l=2
        sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2
        sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2
        sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2
        sh_bands[:, 8] = (extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 2] * extrinsics[:, 2]) * coff_2
        return sh_bands
    
    def forward(self, uv_map, view_dir, light_dir):
        """
        Args:
        - uv_map: B × 3 × H × W
        - mask: B × 3 × H × W
        - view_dir: the position of the camera
        - light_dir: 
        """
        uv_map = uv_map.permute(0, 2, 3, 1)[:, :, :, :2]            # B × H × W × 2

        x = self.texture(uv_map)                                    # B × C × H × W

        basis = self._spherical_harmonics_basis(light_dir)
        basis = basis.to(self.device)
        basis = basis.view(basis.shape[0], basis.shape[1], 1, 1)    # B × 9 × 1 × 1
        x[:, 3:12, :, :] = x[:, 3:12, :, :] * basis

        y = self.generator(x)

        return x, y