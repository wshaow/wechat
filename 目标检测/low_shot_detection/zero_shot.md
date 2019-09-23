## <center>Zero-Shot Object Detection</center>

### 创新点

1）Baseline ZSD Model 提出的基准模型

2）Background-Aware Zero-Shot Detection 对于背景类的两种处理方法

3）Densely Sampled Embedding Space 对样本数据的扩充

### 提出的方法

#### 1、Baseline ZSD Model

![zero_shot](http://pwfic6399.bkt.clouddn.com/paper/zero_shot.PNG?imageView2/0/q/75|watermark/2/text/d3NoYW93/font/YXJpYWw=/fontsize/400/fill/I0NBQkFDQQ==/dissolve/73/gravity/SouthEast/dx/10/dy/10|imageslim)

从整张图片上提取obejctness proposal regions.将这些regions wrap到224*224大小，经过特征提取backbone (论文中使用的Inception-Resnet v2)获得的图像特征。将这些图像特征映射到300维的语义特征空间。通过相似性比较策略获得预测的类别。(根据作者的描述，这里用的方法应该是SAE方法)

$$ {\psi}_i = W_{p} {\phi}(b_i)$$

$b_i$ 是检测出的目标候选框，$W_p$ 是映射矩阵，$\phi$用于提取深度特征。

对于Loss:

$${\cal L}(b_i, y_i, \theta) = {\sum}_{j \in {\cal S},j \ne i} \max (0, m-S_{ii} + S_{ij})$$

预测：

$${\hat y}_i = \mathop{\arg\max}_{j \in {\cal U}} S_{ij}$$

#### 2、Background-Aware Zero-Shot Detection 

1) Statically Assigned Background (SB) Based Zero-Shot Detection

背景类，对应一个固定的标签向量[1, ..., 0]

2) Latent Assignment Based (LAB) Zero-Shot Detection

因为背景可能包含未知类别猜想隐含数，基于观测的数据和猜测的隐含数一起最大化似然函数。之后重复这个过程，这就是EM的思路。作者的做法是：构建了一个不包含seen和unseen类别的单词列表C，先使用baseline ZSD方法预测一些背景的类别，给部分背景框加上标签后添加到数据集中进行下一轮的训练。这样重复五次得到最终的结果。

![bg算法](http://pwfic6399.bkt.clouddn.com/paper/bg_suanfa.PNG?imageView2/0/q/75|watermark/2/text/d3NoYW93/font/YXJpYWw=/fontsize/400/fill/I0NBQkFDQQ==/dissolve/73/gravity/SouthEast/dx/10/dy/10|imageslim)

#### 3、Densely Sampled Embedding Space

为了增加标签的多样性，将OI数据集中未知类去掉之后加到MSCOCO和VG数据集中，增加已知类别的种类，一般来说这样在做特征空间和语义空间之间的映射的时候更加准确一些。



### 实验

#### 1、评估方法

1） 定性评估

![](http://pwfic6399.bkt.clouddn.com/paper/zero_shot_shijueeva.PNG)

2） 定量评估

这个就和 一般的目标检测方法一致



### 总结



#### 需要解决的问题

1）识别结果的分级关系，比如把猫识别成了动物。你不能完全说这个是错的。

2）新目标的bounding box回归需要更加精确。



更多关于编程和机器学习资料请关注FlyAI公众号。
![公众号二维码][1]

[1]: http://pwfic6399.bkt.clouddn.com/wechat/%E5%BE%AE%E4%BF%A1%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.jpg