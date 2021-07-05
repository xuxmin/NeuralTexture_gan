from core.utils.osutils import join
from core.utils.osutils import exists
from core.utils.osutils import isdir
from core.utils.osutils import mkdir_p
from core.utils.osutils import dirname
from shutil import copyfile

OBJECT_NAME = "tissue"
DATA_ROOT = "D:\\Code\\Project\\NeuralTexture_gan\\data"
TARGET_PATH = join(DATA_ROOT, OBJECT_NAME, '.neural_texture')

light_path = join(DATA_ROOT, "lights_8x8.bin")
normal_path = join(DATA_ROOT, OBJECT_NAME, 'gt', '0', 'result256', 'normal_geo_gloabl.exr')


if not isdir(TARGET_PATH):
    mkdir_p(TARGET_PATH)

try:
    # copy light data
    copyfile(light_path, join(TARGET_PATH, "lights_8x8.bin"))

    # copy normal map
    copyfile(normal_path, join(TARGET_PATH, 'normal_geo_gloabl.exr'))

    # copy mask image
    for i in range(24):
        mask_path = join(DATA_ROOT, OBJECT_NAME, 'stereo', str(i), 'mask_cam00.png')
        mask_target_path = join(TARGET_PATH, 'stereo', str(i), 'mask_cam00.png')

        if not isdir(dirname(mask_target_path)):
            mkdir_p(dirname(mask_target_path))

        copyfile(mask_path, mask_target_path)

    print("Copy data done!")
    
except:
    print("Unexpected error")


root = "D:\\Code\\Project\\NeuralTexture_gan\\data\\tissue"
cache_root = join(root, '.neural_texture')

data = join(root, "gt", '0', "img{:0>5d}_cam00.exr".format(1))

tmp = data.replace(root, cache_root).replace('.exr', '.pt')

print(tmp)