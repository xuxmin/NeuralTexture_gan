from pickle import NONE
import cv2
import math
import matplotlib.pyplot as plt
import torch
import imageio
import numpy as np
from core.config import configs, update_config
from core.datasets.egg import EggDataset
from core.datasets.base_dataset import BaseDataset
from core.datasets.bottle import BottleDataset
from core.utils.imutils import imshow
from core.utils.imutils import resize
from core.utils.imutils import show_img_list
from core.utils.imutils import im_to_numpy
from core.utils.imutils import CEToneMapping
from core.utils.imutils import ACESToneMapping
from core.models.pipeline import PipelineModel
from core.models.pipeline import NewGanPipelineModel
from core.models.lp_generator import LPGenerator
import torch
import torch.nn.functional as F


def load_model(model, resume_file):
    checkpoint = torch.load(resume_file)
    print ("=> load checkpoint {}".format(resume_file))
    print ("epoch:", checkpoint['epoch'])
    state_dict_old = checkpoint['state_dict']
    model.load_state_dict(state_dict_old)
    return model


def load_gan_model(model, resume_file):
    checkpoint = torch.load(resume_file)
    print ("=> load checkpoint {}".format(resume_file))
    print ("epoch:", checkpoint['epoch'])
    state_dict_old_G = checkpoint['state_dict_G']
    model.load_state_dict(state_dict_old_G)
    return model


def get_model(checkpoint_path, file_name='model_best.pth.tar'):
    resume_file = "{}\\{}".format(checkpoint_path, file_name)
    # resume_file = "{}\\checkpoint.pth.tar".format(checkpoint_path)
    device = 'cuda'
    # load model
    if 'newganpipeline' in resume_file:
        model = NewGanPipelineModel(256, 256, 16, device).to(device)
        model = load_gan_model(model, resume_file)
        model.eval()
    elif 'LPpipeline' in resume_file:
        model = LPGenerator(256, 256, 16, device).to(device)
        model = load_gan_model(model, resume_file)
        model.eval()
    else:
        model = PipelineModel(256, 256, 16, device).to(device)
        model = load_model(model, resume_file)
        model.eval()
    return model


def valid_model(model, uv_map, normal, view_dir, light_pos, is_lp=False):
    device = 'cuda'
    uv_map = uv_map.unsqueeze(0).to(device)
    normal = normal.unsqueeze(0).to(device)
    view_dir = view_dir.unsqueeze(0).to(device)
    if type(light_pos) == np.ndarray:
        light_pos = torch.from_numpy(light_pos).unsqueeze(0).to(device)
    else:
        light_pos = light_pos.unsqueeze(0).to(device)

    # eval a data
    if is_lp:
        _, output = model.generate(uv_map, normal, view_dir, light_pos)
    else:
        _, output = model(uv_map, normal, view_dir, light_pos)

    return output[0]


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


def gen_video(name, model1, model2):
    """
    要想成功生成视频，注意
    - 尺寸不能是任意的, 可以用 (640, 480)
    - 图像的格式也有规定, 看下面吧
    """
    eggDataset_valid = BottleDataset(configs.DATASET.ROOT, is_train=True)

    configs.DATASET.DLV = False
    uv_map, gt, mask, normal, light_pos1, view_dir1 = eggDataset_valid[100]

    fps = 24.0            # 视频帧率
    size = (640, 480)   
    # size = (1290, 720)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    video = cv2.VideoWriter(name, fourcc, fps, size)
    
    light_origin = np.asarray([252.81884766, -302.20294189,  377.54278564])


    for i in range(360):
        rot_angle = math.radians(i)
        transform_matrix = np.array(
            [
                [ math.cos(rot_angle), math.sin(rot_angle), 0],
                [-math.sin(rot_angle), math.cos(rot_angle), 0],
                [ 0, 0, 1]
            ]
        )
        light_position = np.matmul(transform_matrix, light_origin)

        light_dir = light_position / np.sqrt(np.sum(light_position * light_position))

        print ("{} light_dir: {} view_dir: {}".format(i, light_dir, view_dir1))

        IMAGE_SIZE = 256
        TONE_MAPPING = 20

        configs.DATASET.DLV = False
        output1 = valid_model(model1, uv_map, normal, view_dir1, light_dir).detach().cpu()
        image1 = torch.exp(output1 * 3) - math.exp(-3)          # image [0, ∞]
        # tone_img1 = CEToneMapping(image1, TONE_MAPPING)
        tone_img1 = ACESToneMapping(image1, TONE_MAPPING)
        tone_img1 = torch.clamp(tone_img1, 0, 1)
        tone_img1 = resize(tone_img1, IMAGE_SIZE, IMAGE_SIZE)
        npimg1 = im_to_numpy(tone_img1*255).astype(np.uint8)[:, :, [2, 1, 0]]

        
        if model2 is not None:
            output2 = valid_model(model2, uv_map, normal, view_dir1, light_dir).detach().cpu()
            image2 = torch.exp(output2 * 3) - math.exp(-3)       # image [0, ∞]
            tone_img2 = CEToneMapping(image2, TONE_MAPPING) * mask
            tone_img2 = resize(tone_img2, IMAGE_SIZE, IMAGE_SIZE)
            npimg2 = im_to_numpy(tone_img2*255).astype(np.uint8)[:, :, [2, 1, 0]]

            image = np.concatenate((npimg1, npimg2), axis=1)
            pad_w = (size[0] - IMAGE_SIZE*2) // 2
            pad_h = (size[1] - IMAGE_SIZE) // 2

        else:
            image = npimg1
            pad_w = (size[0] - IMAGE_SIZE) // 2
            pad_h = (size[1] - IMAGE_SIZE) // 2

        pad_img = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        cv2.imwrite("output/temp/{}.png".format(i), pad_img)
        temp = cv2.imread("output/temp/{}.png".format(i))

        video.write(temp)

    video.release()
    cv2.destroyAllWindows()


configs.DATASET.ROOT = "D:\\Code\\Project\\NeuralTexture_gan\\data\\bottle_bak"
configs.DATASET.MODE = "ALL_DATA"
configs.TRAIN.PROCESS = False

# alldata 7/8
# checkpoint =  "D:\\Code\\Project\\NeuralTexture_gan\\log\\fabric\\newganpipeline\\newganpipeline_tex256_f16_alldata_loss1-10-10-10"
checkpoint =  "D:\\Code\\Project\\NeuralTexture_gan\\log\\bottle_bak\\newganpipeline\\newganpipeline_tex256_f16_alldata_loss1-10-10-10_augment"
# checkpoint_l1 =  "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\newganpipeline_final\\newganpipeline_tex256_f16_alldata_loss0-1-0-0"


model1 = get_model(checkpoint, 'checkpoint.pth.tar')
# model2 = get_model(checkpoint_l1, 'checkpoint.pth.tar')


gen_video("output/test_bottle_bak.avi", model1, None)


