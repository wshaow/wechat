# <cneter> Classcial Object Detection Methods </center>

[TOC]



## RCNN系列

### [Faster R-CNN](https://arxiv.org/abs/1506.01497)

#### 网络框架

![faster RCNN框架](https://pic3.zhimg.com/80/v2-c0172be282021a1029f7b72b51079ffe_hd.jpg)

**主要分为4个部分**：

> - Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。
> - Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。
> - Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。
> - Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。



一个更加具体的网络结构：

![](https://pic4.zhimg.com/80/v2-e64a99b38f411c337f538eb5f093bdf3_hd.jpg)

首先先将一个任意大小的图像进行resize到指定的初始输入图像的大小（N*M）。然后再经过一个Conv layers， 得到一个Feature Map。这里我比较迷惑的的地方是如何得到Proposal的以及Proposal得到的是什么？ROIPooling具体做了什么事，得到的结果是什么？

#### Region Proposal Networks(RPN)

![RPN网络结构](https://pic3.zhimg.com/80/v2-1908feeaba591d28bee3c4a754cca282_hd.jpg)



 上图展示了RPN网络的具体结构。可以看到RPN网络实际分为2条线，上面一条通过softmax分类anchors获得positive和negative分类（也就是物体还是背景的二分类问题），下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。 

**这里需要说一下anchor**：

![](https://pic1.zhimg.com/80/v2-c93db71cc8f4f4fd8cfb4ef2e2cef4f4_hd.jpg)

这个在一文读懂faster RCNN中写的比较清楚，总之就是在特征图中以每个点为中心框出一些列的框出来，这些框就是anchors。 遍历Conv layers计算获得的feature maps，为每一个点都配备这9种anchors作为初始的检测框。这样做获得检测框很不准确，不用担心，后面还有2次bounding box regression可以修正检测框位置。 

![anchors](https://pic4.zhimg.com/v2-1ab4b6c3dd607a5035b5203c76b078f3_r.jpg)

 可以看到其num_output=18，也就是经过该卷积的输出图像为WxHx18大小（注意第二章开头提到的卷积计算方式）。这也就刚好对应了feature maps每一个点都有9个anchors，同时每个anchors又有可能是positive和negative，所有这些信息都保存WxHx(9*2)大小的矩阵。为何这样做？后面接softmax分类获得positive anchors，也就相当于初步提取了检测目标候选区域box（一般认为目标在positive anchors中）。  那么为何要在softmax前后都接一个reshape layer？其实只是为了便于softmax分类，至于具体原因这就要从caffe的实现形式说起了。 

 **其实RPN最终就是在原图尺度上，设置了密密麻麻的候选Anchor。然后用cnn去判断哪些Anchor是里面有目标的positive anchor，哪些是没目标的negative anchor。所以，仅仅是个二分类而已！** 



#### 对proposals进行bounding box regression

![](https://pic4.zhimg.com/80/v2-93021a3c03d66456150efa1da95416d3_hd.jpg)

![](http://wshaow.club/paper/fasterrcnn_boundingboxreg.jpg)

![](https://pic3.zhimg.com/v2-8241c8076d60156248916fe2f1a5674a_r.jpg)

####  Proposal Layer

 Proposal Layer有3个输入：positive vs negative anchors分类器结果rpn_cls_prob_reshape，对应的bbox reg的$[d_x(A), d_y(A), d_w(A), d_h(A)]$变换量rpn_bbox_pred，以及im_info；另外还有参数feat_stride=16。
首先解释im_info。对于一副任意大小PxQ图像，传入Faster RCNN前首先reshape到固定MxN，im_info=[M, N, scale_factor]则保存了此次缩放的所有信息。然后经过Conv Layers，经过4次pooling变为WxH=(M/16)x(N/16)大小，其中feature_stride=16则保存了该信息，用于计算anchor偏移量。 

Proposal Layer forward（caffe layer的前传函数）按照以下顺序依次处理：

1. 生成anchors，利用$[d_x(A), d_y(A), d_w(A), d_h(A)]$对所有的anchors做bbox regression回归（这里的anchors生成和训练时完全一致）
2. 按照输入的positive softmax scores由大到小排序anchors，提取前pre_nms_topN(e.g. 6000)个anchors，即提取修正位置后的positive anchors。
3. 限定超出图像边界的positive anchors为图像边界（防止后续roi pooling时proposal超出图像边界）
4. 剔除非常小（width<threshold or height<threshold）的positive anchors
5. 进行nonmaximum suppression
6. Proposal Layer有3个输入：positive和negative anchors分类器结果rpn_cls_prob_reshape，对应的bbox reg的(e.g. 300)结果作为proposal输出。

之后输出proposal=[x1, y1, x2, y2]，注意，由于在第三步中将anchors映射回原图判断是否超出边界，所以这里输出的proposal是对应MxN输入图像尺度的，这点在后续网络中有用。另外我认为，严格意义上的检测应该到此就结束了，后续部分应该属于识别了。

RPN网络结构就介绍到这里，总结起来就是：
**生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals**

#### RoI pooling

而RoI Pooling层则负责收集proposal，并计算出proposal feature maps，送入后续网络。Rol pooling层有2个输入：

1. 原始的feature maps
2. RPN输出的proposal boxes（大小各不相同）

**RoI Pooling原理**

其中有新参数pooled_w和pooled_h，另外一个参数spatial_scale认真阅读的读者肯定已经知道知道用途。RoI Pooling layer forward过程：

- 由于proposal是对应MXN尺度的，所以首先使用spatial_scale参数将其映射回(M/16)X(N/16)大小的feature map尺度；
- 再将每个proposal对应的feature map区域水平分为 pooled_w x pooled_h的网格；
- 对网格的每一份都进行max pooling处理。

这样处理后，即使大小不同的proposal输出结果都是 pooled_w x pooled_h 固定大小，实现了固定长度输出。

![](https://pic1.zhimg.com/80/v2-e3108dc5cdd76b871e21a4cb64001b5c_hd.jpg)



#### 训练

![](https://pic2.zhimg.com/v2-c39aef1d06e08e4e0cec96b10f50a779_r.jpg)

![](https://www.zhihu.com/equation?tex=%5Ctext%7BL%7D%28%5C%7Bp_i%5C%7D%2C%5C%7Bt_i%5C%7D%29%3D%5Cfrac%7B1%7D%7BN_%7B%5Ctext%7Bcls%7D%7D%7D%5Csum_%7Bi%7D%5Ctext%7BL%7D_%5Ctext%7Bcls%7D%28p_i%2Cp_i%5E%2A%29%2B%5Clambda%5Cfrac%7B1%7D%7BN_%7B%5Ctext%7Breg%7D%7D%7D%5Csum_%7Bi%7Dp_i%5E%2A%5Ctext%7BL%7D_%5Ctext%7Breg%7D%28t_i%2Ct_i%5E%2A%29%5C%5C)

这个还是直接看[原文](https://zhuanlan.zhihu.com/p/31426458)

#### 对RCNN和fastRCNN的总结

fast RCNN 和 RCNN都使用了传统的Region Proposal的方法。fast RCNN 是在特征图上通过ROIPooling层得到Proposal区域的特征（特征图的位置可以和原始图像的位置进行映射），ROIPooling层做的事应该就是把对应的Proposal区域的特征池化到相同大小，送入后面的分类和回归网络。

![fast RCNN图](https://mmbiz.qpic.cn/mmbiz_png/4lN1XOZshfct2xp5D6kHjZdKcWhRMLlyQCbdOM19Eo8d7SgrfSxWQc4u5iaK0SkOqHDaJIHl8MSkGDlAKpUkjYg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



## [SSD](https://arxiv.org/abs/1512.02325)

#### 网络结构

![](https://pic1.zhimg.com/v2-a43295a3e146008b2131b160eec09cd4_r.jpg)

 主要思路是均匀地在图片的不同位置进行密集抽样，抽样时可以采用不同尺度和长宽比，然后利用CNN提取特征后直接进行分类与回归，整个过程只需要一步，所以其优势是速度快，但是均匀的密集采样的一个重要缺点是训练比较困难，这主要是因为正样本与负样本（背景）极其不均衡 （ [Focal Loss](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1708.02002) 就是针对这个问题提出来的）。

这里面conv6使用了空洞卷积，关于空洞卷积可以[参考文献](https://arxiv.org/pdf/1511.07122.pdf)或者[知乎文章](https://www.zhihu.com/question/54149221)

#### 设置先验框

这里的先验框其实和faster RCNN中的anchor的含义是差不多的。

![](https://pic1.zhimg.com/v2-f6563d6d5a6cf6caf037e6d5c60b7910_r.jpg)

### 检测

 SSD直接采用卷积对不同的特征图来进行提取检测结果。对于形状为 ![[公式]](https://www.zhihu.com/equation?tex=m%5Ctimes+n+%5Ctimes+p) 的特征图，只需要采用 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes+3+%5Ctimes+p) 这样比较小的卷积核得到检测值。  对于每个单元的每个先验框，其都输出一套独立的检测值，对应一个边界框，主要分为两个部分。第一部分是各个类别的置信度或者评分，值得注意的是SSD将背景也当做了一个特殊的类别，如果检测目标共有 ![[公式]](https://www.zhihu.com/equation?tex=c) 个类别，SSD其实需要预测 ![[公式]](https://www.zhihu.com/equation?tex=c%2B1) 个置信度值，其中第一个置信度指的是不含目标或者属于背景的评分。后面当我们说 ![[公式]](https://www.zhihu.com/equation?tex=c) 个类别置信度时，请记住里面包含背景那个特殊的类别，即真实的检测类别只有 ![[公式]](https://www.zhihu.com/equation?tex=c-1) 个。在预测过程中，置信度最高的那个类别就是边界框所属的类别，特别地，当第一个置信度值最高时，表示边界框中并不包含目标。第二部分就是边界框的location，包含4个值 ![[公式]](https://www.zhihu.com/equation?tex=%28cx%2C+cy%2C+w%2C+h%29) ，分别表示边界框的中心坐标以及宽高。 

 从后面新增的卷积层中提取Conv7，Conv8_2，Conv9_2，Conv10_2，Conv11_2作为检测所用的特征图，加上Conv4_3层，共提取了6个特征图，其大小分别是 ![[公式]](https://www.zhihu.com/equation?tex=%2838%2C+38%29%2C+%2819%2C+19%29%2C+%2810%2C+10%29%2C+%285%2C+5%29%2C+%283%2C+3%29%2C+%281%2C+1%29) ，但是不同特征图设置的先验框数目不同（同一个特征图上每个单元设置的先验框是相同的，这里的数目指的是一个单元的先验框数目）。先验框的设置，包括尺度（或者说大小）和长宽比两个方面。对于先验框的尺度，其遵守一个线性递增规则：随着特征图大小降低，先验框尺度线性增加： 

![](https://www.zhihu.com/equation?tex=s_k+%3D+s_%7Bmin%7D+%2B+%5Cfrac%7Bs_%7Bmax%7D+-+s_%7Bmin%7D%7D%7Bm-1%7D%28k-1%29%2C+k%5Cin%5B1%2Cm%5D)

## [YOLO](https://arxiv.org/abs/1804.02767)

这个主要关注一下它的损失函数就好。

框架：

![](https://img1.mukewang.com/5b2369a10001ec1812200534.jpg)



损失函数：

![](https://img1.mukewang.com/5b2369a20001078908910615.jpg)

上图有错误，20对应的应该是类别的类似一个softmax的输出

## 经典目标检测性能

![](https://pic2.zhimg.com/80/v2-f143b28b7a7a1f912caa9a99c1511849_hd.jpg)

## [Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf)

#### 总体框架

![](https://pic1.zhimg.com/v2-7a539f4d5f904db3c4559ebe6c9ef49c_r.jpg)

#### ROI Align

在Faster RCNN中，有两次整数化的过程：

1. region proposal的xywh通常是小数，但是为了方便操作会把它整数化。
2. 将整数化后的边界区域平均分割成 k x k 个单元，对每一个单元的边界进行整数化。

两次整数话过程为：

![](https://pic3.zhimg.com/80/v2-36e08338390289e0dd88203b4e8ddda2_hd.jpg)

 事实上，经过上述两次整数化，此时的候选框已经和最开始回归出来的位置有一定的偏差，这个偏差会影响检测或者分割的准确度。在论文里，作者把它总结为“不匹配问题”（misalignment）。 

为了解决这个问题，ROI Align方法取消整数化操作，保留了小数，使用以上介绍的双线性插值的方法获得坐标为浮点数的像素点上的图像数值。但在实际操作中，ROI Align并不是简单地补充出候选区域边界上的坐标点，然后进行池化，而是重新进行设计。

下面通过一个例子来讲解ROI Align操作。如下图所示，虚线部分表示feature map，实线表示ROI，这里将ROI切分成2x2的单元格。如果采样点数是4，那我们首先将每个单元格子均分成四个小方格（如红色线所示），每个小方格中心就是采样点。这些采样点的坐标通常是浮点数，所以需要对采样点像素进行双线性插值（如四个箭头所示），就可以得到该像素点的值了。然后对每个单元格内的四个采样点进行maxpooling，就可以得到最终的ROIAlign的结果。

![](https://pic1.zhimg.com/v2-76b8a15c735d560f73718581de34249c_r.jpg)



## [FPN](https://arxiv.org/abs/1612.03144)



## reference

[一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)

[目标检测|SSD原理与实现](https://zhuanlan.zhihu.com/p/33544892)

[深入理解目标检测与YOLO（从v1到v3）](https://www.imooc.com/article/36391)

[令人拍案称奇的Mask RCNN](https://zhuanlan.zhihu.com/p/37998710)