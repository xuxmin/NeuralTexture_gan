import cv2
import math
import matplotlib.pyplot as plt
import torch
import imageio
import numpy as np
from core.config import configs, update_config
from core.datasets.egg import EggDataset
from core.utils.imutils import imshow
from core.utils.imutils import show_img_list
from core.utils.imutils import im_to_numpy
from core.utils.imutils import CEToneMapping
from core.models.pipeline import PipelineModel
from core.models.pipeline import NewGanPipelineModel
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
    else:
        model = PipelineModel(256, 256, 16, device).to(device)
        model = load_model(model, resume_file)
        model.eval()
    return model


def valid_model(model, uv_map, normal, view_dir, light_pos):
    device = 'cuda'
    uv_map = uv_map.unsqueeze(0).to(device)
    normal = normal.unsqueeze(0).to(device)
    view_dir = view_dir.unsqueeze(0).to(device)
    if type(light_pos) == np.ndarray:
        light_pos = torch.from_numpy(light_pos).unsqueeze(0).to(device)
    else:
        light_pos = light_pos.unsqueeze(0).to(device)

    # eval a data
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
    eggDataset_valid = EggDataset(configs.DATASET.ROOT, is_train=False)

    configs.DATASET.DLV = False
    uv_map, gt, mask, normal, light_pos1, view_dir1 = eggDataset_valid.getData(7, 282)

    configs.DATASET.DLV = True
    uv_map, gt, mask, normal, light_pos2, view_dir2 = eggDataset_valid.getData(7, 282)


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

        configs.DATASET.DLV = False
        output1 = valid_model(model1, uv_map, normal, view_dir1, light_dir)

        light_dir_map = eggDataset_valid.getLightDirMap(7, 282, light_position)

        sample = uv_map.permute(1, 2, 0)[:, :, :2].unsqueeze(0)            # 1 × H × W × 2
        sample = 2 * sample - 1
        sample[:, :, :, 1] = -sample[:, :, :, 1]

        light_dir = sample_texture(light_dir_map, sample)

        configs.DATASET.DLV = True
        output2 = valid_model(model2, uv_map, normal, view_dir2, light_dir)
        # output3 = valid_model(model3, uv_map, normal, view_dir, light_dir)

        image1 = torch.exp(output1.detach().cpu() * 3) - math.exp(-3)         # image [0, ∞]
        image2 = torch.exp(output2.detach().cpu() * 3) - math.exp(-3)         # image [0, ∞]
        # image3 = torch.exp(output3.detach().cpu() * 3) - math.exp(-3)         # image [0, ∞]
        # image[mask == 0] = 0

        tone_img1 = CEToneMapping(image1, 3)
        tone_img2 = CEToneMapping(image2, 3)
        # tone_img3 = CEToneMapping(image3, 3)

        npimg1 = im_to_numpy(tone_img1*255).astype(np.uint8)[:, :, [2, 1, 0]]
        npimg2 = im_to_numpy(tone_img2*255).astype(np.uint8)[:, :, [2, 1, 0]]
        # npimg3 = im_to_numpy(tone_img3*255).astype(np.uint8)[:, :, [2, 1, 0]]

        # npimg1 = im_to_numpy(img1)[:, :, [2, 1, 0]]
        # npimg2 = im_to_numpy(img2)[:, :, [2, 1, 0]]
        # image = np.concatenate((npimg1, npimg2), axis=1) 
        image = npimg2
        # image = npimg

        pad_w = (size[0] - 256) // 2
        pad_h = (size[1] - 256) // 2
        pad_img = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        cv2.imwrite("output/temp/{}.png".format(i), pad_img)
        temp = cv2.imread("output/temp/{}.png".format(i))

        # cv2.putText(temp, "gan", (185, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # cv2.putText(temp, "no gan", (435, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # cv2.putText(temp, "Epoch {}".format(10), (870, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        

        video.write(temp)

    video.release()
    cv2.destroyAllWindows()


configs.DATASET.ROOT = "D:\\Code\\Project\\NeuralTexture_gan\\data"
configs.DATASET.MODE = "ALL_DATA_7/8"
configs.TRAIN.PROCESS = False

checkpoint_path = "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\pipeline\\pipeline_batch4_adam_lr1e-3_tex256_f16_testmore"
checkpoint_path_gan1 = "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\newganpipeline\\newganpipeline_batch1_SGDAdam_lr1e-3_tex256_f16_testmore_loss1-10-10-10"
checkpoint_path_gan2= "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\newganpipeline\\newganpipeline_batch1_SGD-8e-1-Adam_lr1e-3_tex256_f16_alldata_loss1-10-10-10"

checkpoint_alldata = "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\newganpipeline\\newganpipeline_batch1_SGD-8e-1-Adam_tex256_f16_alldata_augment_debug"

# test_more
checkpoint_test_more = "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\newganpipeline\\newganpipeline_batch1_SGD-8e-1-Adam_tex256_f16_testmore_augment_debug"

# concate
checkpoint_concate = "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\newganpipeline\\newganpipeline_batch1_SGD-8e-1-Adam_tex256_f16_testmore_augment_debug_concate"

# alldata
checkpoint_alldata = "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\newganpipeline\\newganpipeline_batch1_SGD-8e-1-Adam_tex256_f16_alldata_augment_debug"

# alldata L1 loss
checkpoint_alldata_l1 =  "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\newganpipeline\\newganpipeline_batch1_Adam_lr1e-3_tex256_f16_alldata_loss-l1_augment_debug"

# dlv
checkpoint_dlv =  "D:\\Code\\Project\\NeuralTexture_gan\\log\\egg\\newganpipeline\\newganpipeline_batch1_SGD-8e-1-Adam_tex256_f16_alldata_augment_debug_dlv"


model1 = get_model(checkpoint_alldata, 'checkpoint_epoch10.pth.tar')
model2 = get_model(checkpoint_dlv, 'checkpoint.pth.tar')
# model3 = get_model(checkpoint_alldata, 'checkpoint_epoch9.pth.tar')


gen_video("output/best_result.avi", model1, model2)


