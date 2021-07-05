
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import random
import torch.nn.functional as F
import torch
from torchvision import transforms as transforms

from .misc import to_torch
from .misc import to_numpy


def im_to_torch(img):
    """
    convert ndarray image to tensor

    Args:
    - img: ndarray, H × W × C

    Returns:
    - tensor: C × H × W, range [0, 1]
    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def im_to_numpy(img):
    """
    convert tensor image to ndarray

    Args:
    - img: tensor, C × H × W

    Returns:
    - img: ndarray,  H × W × C
    """
    if type(img) is np.ndarray:
        return img
    img = to_numpy(img)
    res_img = np.transpose(img, (1, 2, 0))  # H*W*C
    return res_img


def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    # print('%f %f' % (img.min(), img.max()))
    img = cv2.resize(
        img,
        (oheight, owidth)
    )
    img = im_to_torch(img)
    # print('%f %f' % (img.min(), img.max()))
    return img


def augment(img_list, output_size):
    """
    img_list: 要同时处理的图片
    """

    crop_size = random.randint(output_size // 10 * 6, output_size // 10 * 9) 
    top = random.randint(0, output_size - crop_size)
    left = random.randint(0, output_size - crop_size)  

    scale_factor = random.randint(70, 110) / 100
    scale_size = int(crop_size * scale_factor)
    scale_size = scale_size if scale_size % 2 == 0 else scale_size + 1

    pad = (output_size - scale_size) // 2       # 确保 output_size 大于 scale_size

    result_img = []
    for img in img_list:
        tmp = im_to_numpy(img)

        crop_img = tmp[top:top+crop_size, left:left+crop_size, :]

        scale_img = cv2.resize(crop_img, (scale_size, scale_size))

        # 感觉不能旋转
        # matRotate = cv2.getRotationMatrix2D((scale_size*0.5, scale_size*0.5), rot_angle, 1)
        # rot_img = cv2.warpAffine(scale_img, matRotate, (scale_size, scale_size))
        
        pad_img = cv2.copyMakeBorder(scale_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        pad_img = im_to_torch(pad_img)

        result_img.append(pad_img)


    #     crop_im = transforms.RandomCrop(crop_size)(img)   # 裁剪出crop_size × crop_size的区域
    #     scale_img = transforms.Resize(crop_size * scale_factor, crop_size * scale_factor)(crop_im)
    #     rot_img = transforms.RandomRotation(rot_angle)(scale_img)        # 随机旋转45度

    #     pad1 = math.abs(output_size[0] - rot_img.size(0)) // 2
    #     pad2 = math.abs(output_size[1] - rot_img.size(1)) // 2

    #     pad_img = transforms.Pad((pad1, pad2))(rot_img)

    #     pad_img = im_to_torch(pad_img)

    #     result_img.append(pad_img)


    return result_img
    

def load_image(path, to_tensor=True):
    """
    read .exr image and convert to ndarray.

    Args:
    - path: .exr image path

    Returns:
    - tensor: C × H × W, range [0, ???] or ndarray: H × W × C range [0, ???]
    """
    # import OpenEXR
    # import Imath
    # import array
    # file = OpenEXR.InputFile(path)
    # # Compute the size
    # dw = file.header()['dataWindow']
    # sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)     # width height
    # # Read the three color channels as 32-bit floats
    # FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # (R, G, B) = [np.array(array.array('f', file.channel(Chan, FLOAT)).tolist(), dtype=np.float32).reshape(sz[1], sz[0]) for Chan in ("R", "G", "B") ]
    # image = np.dstack((R, G, B))
    
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    image = image[:, :, [2, 1, 0]]

    if to_tensor:
        data = im_to_torch(image)
    else:
        data = image
    return data


def load_png(path, to_tensor=True):
    image = cv2.imread(path)
    if np.max(image) > 1:
        image = image / 255
    if to_tensor:
        return im_to_torch(image)
    return image


# tone mapping, see more: https://zhuanlan.zhihu.com/p/21983679
def CEToneMapping(img, adapted_lum):
    return 1 - np.exp(-adapted_lum * img)


def Uncharted2ToneMapping(img, adapted_lum):
    def F(x):
        A = 0.22
        B = 0.30
        C = 0.10
        D = 0.20
        E = 0.01
        F = 0.30
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    WHITE = 11.2
    return F(1.6 * adapted_lum * img) / F(WHITE)


def ACESToneMapping(img, adapted_lum):
	A = 2.51
	B = 0.03
	C = 2.43
	D = 0.59
	E = 0.14

	img *= adapted_lum
	return (img * (A * img + B)) / (img * (C * img + D) + E)


def imshow(img, tone_mapping=True):
    """
    display image with tone mapping
    
    Args:
    - img: tensor C × H × W, range [0, 1], or ndarray H × W × C
    """
    fig=plt.figure()
    tmp = img
    if tone_mapping:
        # tmp = CEToneMapping(tmp, 10)
        tmp = ACESToneMapping(tmp, 10)

    npimg = im_to_numpy(tmp * 255).astype(np.uint8)
    plt.imshow(npimg)
    plt.axis('off')
    plt.show()


def show_img_list(img_list, size, rol_col, tone_mapping=True, title_list = None):
    num = len(img_list)
    fig=plt.figure(figsize=size)
    for i in range(1, num+1):
        tmp = img_list[i-1]
        if tone_mapping:
            # tmp = CEToneMapping(tmp, 10)
            tmp = ACESToneMapping(tmp, 10)
        npimg = im_to_numpy(tmp * 255).astype(np.uint8)
        fig.add_subplot(rol_col[0], rol_col[1], i)
        if title_list:
            plt.title(title_list[i-1], y=-0.1)
        plt.imshow(npimg)
        plt.axis('off')
    plt.show()


def imwrite(filename, img, tone_mapping=True):
    """
    save an exr image
    
    Args:
    - img: tensor C × H × W, range [0, 1], or ndarray H × W × C
    """
    if tone_mapping:
        tone_img = CEToneMapping(img, 3)
    else:
        tone_img = img
    npimg = im_to_numpy(tone_img * 255).astype(np.uint8)
    
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)

    cv2.imwrite(filename, npimg)


def sample_texture(texture, sample):
    """
    texture: 1 × 3 × H × W
    sample: 1 × H × W × 2       range [-1, 1]

    Return: 3 × H × W
    """
    texture = texture.unsqueeze(0)
    texture_R = F.grid_sample(texture[:, 0:1, :, :], sample, align_corners=True)
    texture_G = F.grid_sample(texture[:, 1:2, :, :], sample, align_corners=True)
    texture_B = F.grid_sample(texture[:, 2:3, :, :], sample, align_corners=True)
    result = torch.cat(tuple([texture_R, texture_G, texture_B]), dim=1)
    return result[0]