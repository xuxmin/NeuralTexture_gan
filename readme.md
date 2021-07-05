1. 一定要加 augment, 否则生成物体动的视频的时候会非常烂
2. 裁剪的时候尽可能大一些，撑满整个图片



整体流程

1. 将数据拷贝到 data/ 中。
    - 检查是否有 stereo/  目录以及对应的 mask_cam00.png 文件
    - 将 normal map lights_8x8.bin 复制到根目录
2. 将模型拷贝到 D:\Code\Project\temp\PathTracer\PathTracer
3. 运行 gen_uv_map.py. 生成 uv_map 文件
    - 需要注意外参与内参文件，如果是同一个相机就行了。
    - 注意修改模型文件
    - 比较生成的 uv_map 与 实拍照片是否对应
4. 编写配置文件
    - 修改 OBJECT_NAME
    - 修改 DATASET.NAME
    - 修改 DATASET.ROOT
    - 修改 xxxx 在 test_model.ipynb 中检查


生成视频

1. 将 模型文件复制到 D:\Code\Project\tools\PathTracer_NEWCAM_ROT 项目中
2. 修改配置文件中的 UV_MAP_X2 ... 



遇到的问题：

增加分辨率到 512 x 512 后，生成的图片某些地方会出现一个洞。。。这真的是。。。是为什么？我记得以前训练蛋的时候也遇到过，不知道是怎么解决的。。。。
- 尝试继续训练，增加augment的力度（裁剪一小块）

训练一开始生成的图片就会有洞，这是为什么？会不会越训练越小呢？暂时发现的确随着训练，洞慢慢变小了，不知道后面会不会消失。。。

会不会是这里过于亮了，像素值非常大，导致出问题？

---

修改一下预处理方式如何？？我分析了一下

图片 384 张中，大部分图片像素最大值大约 0.2，10张左右图片像素最大值会大于1，有的会有 12.

图片的平均像素值为 0.02 左右

原来的初始化方法： math.log((math.exp(-3) +  0.02)) / 3

math.log((math.exp(-3) +  0.02)) / 3 = -0.88
math.log((math.exp(-3) +  0.2)) / 3 = -0.46

导致大部分像素都是负值。。。

不是这个原因

---

修改了损失函数后发现。。与损失函数有关。

判别器的特征匹配损失非常重要，非常有问题。。。。。。。。。。

去掉 FM 与 VGG loss。还是有洞。。

---

先训练久一点看一下吧。。。训练到 29 epoch，感觉还是有一点。。。

---

gt 转成 ldr 后训练试试看。就训练了一点点，发现一开始还是有洞

---

把生成器的初始学习率从 0.0005 调整到 0.0001，不会这么解决了吧.


将初始学习率调低了以后，一开始生成的图片没有洞了，这应该正常了吧。

epoch 0:

<img src="http://182.92.112.211:8080/images/2021/07/03/pred.png" width=300>

<img src="http://182.92.112.211:8080/images/2021/07/03/epoch1.png" width=300>
