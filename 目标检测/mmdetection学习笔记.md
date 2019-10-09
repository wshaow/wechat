## <center>mmdetection学习笔记</center>

### train.py文件

命令行参数解析

从配置文件中读取配置项

关于下面这个的作用：

```python
torch.backends.cudnn.benchmark
```

> It enables benchmark mode in cudnn.
> benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
> But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.
>
> 以上是pytorch官方论坛给出的解释，就是说如果你的输入的大小不是频繁变动，那么cudnn会去根据你输入的大小自动寻找一个自适应算法进行加速。但是如果输入大小是频繁变动的那么cudnn每次都要去寻找对应的最优的算法，这样反而会更加耗时。

通过命令行参数（CLI）更新配置

使用 linear scaling rule配置自动更新学习率

job launcher 是指分布式训练的任务启动器（job launcher），默认值为`none`表示不进行分布式训练。看是否使用分布式训练。

初始化日志打印器

设置随机种子，将numpy torch torch.cuda版本的随机种子设置为一个定值，这样每次运行的随机数是一致的。

通过配置项中的模型参数，构建模型

通过配置项中的参数，构建数据集

设置checkpoint_config

调用训练函数训练模型

### mmdet.api.train.py

在上面的训练文件中调用了train_detector函数

这个函数内部实际上根据是否使用分布式训练又选择性的调用了分布式和非分布式的训练接口。

先看一下非分布式的训练接口`_non_dist_train`

> 先构建dataloader
>
> 将模型参数放到gpu上
>
> 构建优化器
>
> 







