import OpenEXR
import Imath
import array
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import random

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
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


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
    """
    top = random.randint(20, 30)
    left = random.randint(20, 30)  
    scale_factor = random.randint(80, 120) / 100

    crop_size = 280
    scale_size = int(crop_size * scale_factor)
    scale_size = scale_size if scale_size % 2 == 0 else scale_size + 1
    pad = (output_size - scale_size) // 2

    result_img = []
    for idx, img in enumerate(img_list):
        tmp = im_to_numpy(img)

        crop_img = tmp[top:top+crop_size, left:left+crop_size, :]

        scale_img = cv2.resize(crop_img, (scale_size, scale_size))
        
        pad_img = cv2.copyMakeBorder(scale_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        pad_img = im_to_torch(pad_img)

        result_img.append(pad_img)
    
    return result_img
    

def load_image(path, to_tensor=True):
    """
    read .exr image and convert to ndarray.

    Args:
    - path: .exr image path

    Returns:
    - tensor: C × H × W, range [0, ???] or ndarray: H × W × C range [0, ???]
    """
    file = OpenEXR.InputFile(path)
    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)     # width height
    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    (R, G, B) = [np.array(array.array('f', file.channel(Chan, FLOAT)).tolist(), dtype=np.float32).reshape(sz[1], sz[0]) for Chan in ("R", "G", "B") ]
    image = np.dstack((R, G, B))
    
    # image = cv2.imread(path)

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
        tmp = CEToneMapping(tmp, 0.2)
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
            tmp = CEToneMapping(tmp, 0.2)
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
        tone_img = CEToneMapping(img, 0.2)
    else:
        tone_img = img
    npimg = im_to_numpy(tone_img * 255).astype(np.uint8)
    cv2.imwrite(filename, npimg)


# def random_crop(img_list, value_list, width, height):
#     """
#     do ramdom crop and padding to size [width, height]
#     """
#     # top = random.randint(0, 200)
#     # left = random.randint(0, 200)
#     top = 100
#     left = 100
#     size = 350

#     result_img = []

#     for idx, img in enumerate(img_list):

#         tmp = im_to_numpy(img)

#         crop_img = tmp[top:top+size, left:left+size, :]

#         pad = (width - size) // 2

#         pad_img = cv2.copyMakeBorder(crop_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value_list[idx])

#         pad_img = im_to_torch(pad_img)

#         result_img.append(pad_img)

#     return result_img