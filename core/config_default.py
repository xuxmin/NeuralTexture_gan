configs = {
    'OUTPUT_DIR': 'output',
    'LOG_DIR': 'log',
    'PRINT_FREQ': 1,

    'NEURAL_TEXTURE': {
        'SIZE': 256,
        'FEATURE_NUM': 16,
        'HIERARCHY': False,                         
    },

    'OBJECT_NAME': None,        # 模型 obj 文件名, PathTracer_NEWCAM, PathTracer 中都要存

    'DATASET': {
        'NAME': 'egg',
        'ROOT': 'D:\\Code\\Porject\\NeuralTexture-master\\data',
        'VIEW_NUM': 24,
        'LIGHT_NUM': 384,
        'MODE': 'TEST_ALL',
        'AUGMENT': False,
        'DLV': False,
        
        'UV_MAP_X': 0,          # UV_MAP 裁剪
        'UV_MAP_Y': 0,
        'UV_MAP_W': 0,
        'UV_MAP_H': 0,
        
        'MASK_X': 0,            # MASK 裁剪
        'MASK_Y': 0,
        'MASK_W': 0,
        'MASK_H': 0,

        'GT_X': 0,              # GT 裁剪
        'GT_Y': 0,
        'GT_W': 0,
        'GT_H': 0,

        'UV_MAP_X2': 0,         # 用于生成视频的 UV_MAP 裁剪 
        'UV_MAP_Y2': 0,
        'UV_MAP_H2': 0,
        'UV_MAP_W2': 0,
    },

    'MODEL': {
        'NAME': 'pipeline',
        'IMAGE_SIZE': [256, 256],
        'GENERATE': 'Pix2Pix',
        'COMBINE': 'multiplicate',
    },

    'TRAIN': {
        'BATCH_SIZE': 1,
        'SHUFFLE': True,
        'LR': 0.001,
        'G_LR': 1000,
        'D_LR': 1000,
        'LR_STEP': [1115, 1118],
        'LR_FACTOR': 0.1,
        'BEGIN_EPOCH': 0,
        'END_EPOCH': 1,
        'PROCESS': True,

        'LAMBDA_GAN': 1,
        'LAMBDA_L1': 10,
        'LAMBDA_FM': 10,
        'LAMBDA_VGG': 10,
    },
    
    'TEST': {
        'BATCH_SIZE': 2,
    },

    'DEBUG': {
        'CHECK_PER_EPOCH': 0
    },
}