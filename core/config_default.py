configs = {
    'OUTPUT_DIR': 'output',
    'LOG_DIR': 'log',
    'PRINT_FREQ': 1,

    'NEURAL_TEXTURE': {
        'SIZE': 256,
        'FEATURE_NUM': 16,
        'HIERARCHY': False,                         
    },

    'DATASET': {
        'NAME': 'egg',
        'ROOT': 'D:\\Code\\Porject\\NeuralTexture-master\\data',
        'MODE': 'TEST_ALL',
        'AUGMENT': False,
        'DLV': False,
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