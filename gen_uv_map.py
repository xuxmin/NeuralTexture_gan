import os
from core.utils.osutils import mkdir_p
from core.utils.osutils import isdir
    
OBJECT_NAME = 'tissue'

oldval = os.getcwd()                                    # 查看当前工作目录
os.chdir('D:\\Code\\Project\\temp\\PathTracer\\PathTracer')   # 修改当前工作目录


for folder in range(24):
    ex_path = "D:\\Code\\Project\\NeuralTexture_gan\\data\\fabric\\gt\\{}\\result256\\extrinsic.yml".format(folder)
    in_path = "obj/test/intrinsic1.yml"
    output_file = "D:\\Code\\Project\\NeuralTexture_gan\\data\\{}\\gt\\{}\\result256\\render_extrinsic.yml.exr".format(OBJECT_NAME, folder)


    if not isdir(os.path.dirname(output_file)):
        mkdir_p(os.path.dirname(output_file))

    os.system('.\\PathTracer_UV.exe obj\\test\\{}.obj uv {} {} {}'.format(OBJECT_NAME, ex_path, in_path, output_file))


# import cv2
# import numpy as np
# image = cv2.imread("D:\\Code\\Project\\NeuralTexture_gan\\data\\bottle_bak\\gt\\0\\img00000_cam00.exr", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

# def CEToneMapping(img, adapted_lum):
#     return 1 - np.exp(-adapted_lum * img)

# print(image)

# image = CEToneMapping(image, 10)

# print(image)

# cv2.imwrite("test.png", image * 255)