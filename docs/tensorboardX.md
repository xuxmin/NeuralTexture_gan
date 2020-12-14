
### tensorboardX 安装

```bash
pip install tensorboardX
```

## 基本使用

导入 SummaryWriter, 定义一个实例, log_dir 指定生成的文件存放目录

```py
from tensorboardXimport SummaryWriter
writer = SummaryWriter(log_dir=tb_log_dir)       
```

- `add_graph(model, input_to_model, verbose=False)`
```py
dump_input = torch.rand((8, 3, 256, 256))
writer.add_graph(model, (dump_input, ), verbose=False)
```
构造模拟数据输入模型, 使用 add_graph 添加模型的结构。

- `add_scalar(tag, scalar_value, global_step=None)` 添加单个标量
```py
writer.add_scalar('train_loss', losses.avg, global_steps)
```

## 查看结果

```bash
tensorboard --logdir logPath --port 8000
```
在浏览器中打开
```
http://localhost:8000/
```