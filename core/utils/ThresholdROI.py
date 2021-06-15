from skimage.filters import threshold_otsu
from skimage import measure
import numpy as np


def ThresholdROI(uv_map, thresh=None):
    image = (uv_map[:, :, :1] + uv_map[:, :, 1:2] + uv_map[:, :, 2:3]) / 3

    # 1. 阈值分割: 使用自定义阈值
    if thresh is None:
        thresh = threshold_otsu(image)

    dst = (image >= thresh)

    # 2. 连通区域标记
    img_bw = dst * 1.0
    label_image = measure.label(img_bw, connectivity=1)

    # 循环得到每一个连通区域属性集
    for region in measure.regionprops(label_image=label_image):
        minr, minc, _, maxr, maxc, _ = region.bbox

    return minr, minc, maxr - minr, maxc - minc