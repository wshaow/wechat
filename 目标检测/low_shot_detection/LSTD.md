# <center>LSTD阅读笔记</center>

## 论文的创新点

1、提出了一个 low shot transfer detection框架，将 SSD 和 faster-RCNN 进行了整合。

2、提出了一个新的正则化的迁移学习框架，the transfer knowledge (TK) and background depression
(BD) regularizations are proposed to leverage object knowledge respectively from source and target domains, in order to further enhance fine-tuning with a few target images。通过利用源和目标域中的物体知识提出了对 transfer knowledge (TK)和background depression的抑制项。

3、效果显著state-of-the-art。

## introduction

**现有的一些解决办法**：

1、获取额外的目标检测图像，这些图像要比较容易标注，比如一张图中只有一个目标因此只需要一个标签就可以了。通过这种方式可以是弱监督学习或者半监督学习方法。（网上有文章说弱监督:有标签没有框框.半监督:有标签,部分有框.  现在还没有看相关的文章无法确认是不是这样）。缺点：However, the performance of these detectors is often limited, because of lacking sufficient supervision on the training images.

2、使用迁移学习。缺点：1）当目标检测的样本集很小时，使用一般的迁移策略将图像分类的预训练结果迁移到目标识别上是不合适的，原因是使用这么小的样本集无法消除这两者的差异性。2）目标检测模型比较复杂，在微调的时候更容易过拟合。3）如果只是使用简单的迁移方法，会忽略迁移前的源域和目标域中的重要的object。

**提出方法**：

1、上面的创新点

2、框架

![architecture](http://pwfic6399.bkt.clouddn.com/wechat/object_detection/LSTD_architecture.png)



为什么要结合 SSD 和 faster-RCNN？这样不是反而增加了模型的复杂性？SSD的优点是什么？faster-RCNN的优点又是什么？

使用了TK和BD还好理解。

## Related Works

**Low-shot Learning**：

受人仅仅需要少量监督样本的启发。现在主要的方向：弱监督，半监督和迁移学习



## Low-Shot Transfer Detector (LSTD)

### Basic Deep Architecture of LSTD

**1、design bounding box regression in the fashion of SSD.**

为什么要选择SSD的bounding box回归的形式。文中给了两点理由：

> 1）this multiple-convolutional-feature design in SSD is suitable to localize objects with various sizes. This can be especially important for low-shot detection, where we lack training samples with size diversity.
>
> 2） More importantly, the regressor in SSD is shared among all object categories, instead of being specific for each category as in Faster RCNN.（faster RCNN中对每个类别的回归器是特定的？）

好处：

> the regression parameters of SSD, which are pretrained on the large-scale source domain, can be re-used as initialization in the different low-shot target domain. This avoids re-initializing bounding box regression randomly, and thus reduces the fine-tuning burdens with only a few images in the target domain.

说白了就是预训练学到的bounding box回归具有泛化能力。



**2.design object classification in the fashion of Faster RCNN.**

这里对faster RCNN进行了一些修改

### Regularized Transfer Learning for LSTD

#### **Background-Depression (BD) Regularization**

通过使用LBD、LSTD能够在对目标物体更加关注的同时抑制背景区域，对少量训练图像的训练尤为重要

#### **Transfer-Knowledge (TK) Regularization.**



![正则化](http://pwfic6399.bkt.clouddn.com/wechat/object_detection/LSDT_figure2.png)



更多关于编程和机器学习资料请关注FlyAI公众号。
![公众号二维码][1]

[1]: http://pwfic6399.bkt.clouddn.com/wechat/%E5%BE%AE%E4%BF%A1%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.jpg











