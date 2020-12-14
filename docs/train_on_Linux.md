## 在Linux 服务器上运行方法

查看当前存在哪些虚拟环境
```bash
conda env list
```

激活虚拟环境
```bash
source activate <env_name>
```
查看虚拟环境中装了哪些包:
```bash
conda list
```

查看GPU是否可用
```bash
>>> import torch
>>> print(torch.cuda.is_available())
True
>>>
```

指定GPU进行训练
```bash
CUDA_VISIBLE_DEVICES=1  python ./example/main.py --arch hg --stack 1 --block 1 --solver adam --epochs 50 --lr 5e-4 --train-batch 8 --test-batch 8 --checkpoint ./checkpoint/larva/hg-s1-b1
```

恢复训练:
```bash
CUDA_VISIBLE_DEVICES=1  python ./example/main.py --arch hg --stack 1 --block 1 --solver adam --epochs 50 --lr 5e-4 --train-batch 8 --test-batch 8 --checkpoint ./checkpoint/larva/hg-s1-b1 --resume ./checkpoint/larva/hg-s1-b1/checkpoint.pth.tar
```

关闭虚拟环境
```bash
deactivate env_name
```

查看 GPU 使用情况:
```bash
nvidia-smi
```