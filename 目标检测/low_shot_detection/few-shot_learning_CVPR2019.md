## <center>Low-Shot Learning in CVPR 2019</center>

### Metric learning methods

#### Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning
作者：南京大学 李文斌 R&L Group

##### 论文框架

![论文框架图](http://wshaow.club/paper/DN4.png-water_mark)



an image-level feature based measure ：个人理解是直接使用端到端的方式训练得到的最后一层的特征。

a local descriptor based image-to-class measure ：个人理解是经过处理后得到的隐含层的特征。

作者说现在普遍的分类都是基于第一类，但是在few shot的情况下，使用第二种能更好的利用样本中的信息。（实验的讨论部分设计实验进行了验证）

> building upon the recent episodic training mechanism, we propose a Deep Nearest Neighbor Neural Network (DN4 in short) and train it in an end-to-end manner. Its key difference from the literature is the replacement of the image-level feature based measure in the final layer by a local descriptor based image-to-class measure. This measure is conducted online via a k-nearest neighbor search over the deep local descriptors of convolutional feature maps.

就是将最后一层的image-level feature based measure 替换为了local descriptor based image-to-class measure使用KNN来搜索。

##### **Metric learning methods**（主要思想应该是信息的相似性度量）

> The latter type adopts a relatively simpler architecture to learn a deep embedding space to transfer representation (knowledge).
>
> 文中提到的一些相关的论文：
>
> [Siamese Neural Network](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
>
> [episodic training mechanism](https://arxiv.org/pdf/1606.04080.pdf)
>
> [Relation Network](https://arxiv.org/abs/1711.06025)

这篇文章的一个观点是之前的方法忽略了最终的分类这一步。

#####  Naive-Bayes Nearest-Neighbor (NBNN)

作者提到这篇文章主要受到 [Naive-Bayes Nearest-Neighbor (NBNN) approach](http://www.wisdom.weizmann.ac.il/~irani/PAPERS/InDefenceOfNN_CVPR08.pdf) 2008年的CVPR。应该是主要考虑了特征分布的相似性。在这篇文章中主要提供了两个主要的观点：

> **First,** summarizing the local features of an image into a compact image-level representation could lose considerable discriminative information. It will not be recoverable when the number of training examples is small. （说到特征的可恢复性，自动编码机是不是更好）
>
> **Second**, in this case, directly using these local features for classification will not work if an image-to-image measure is used. Instead, an image-to-class measure
> should be taken, by exploiting the fact that a new image can be roughly “composed” using the pieces of other images in the same class.（类似稀疏的观点）

#### Few-Shot Learning with Localization in Realistic Settings

**作者**：Davis Wertheimer 康奈尔大学

这篇文章将few shot 和 样本不均衡结合起来了，也就是正常情况样本分布应该是一个heavy-tailed class distributions（重尾分布）

![重尾分布](http://wshaow.club/paper/heavy_tailed_distrub.png-water_mark)

##### 创新点

> (a) better training procedures based on adapting cross-validation to meta-learning；
> (b) novel architectures that localize objects using limited bounding box annotations before classification；
> (c) simple parameter-free expansions of the feature space based on bilinear pooling.

主要还是开启了一个识别问题的新的研究点。

##### 现有的few shot存在的问题

1、假定每个训练样本的类别是均衡的，但是实际上是重尾分布的；

2、基本上分类的类别很少且类别相差比较大。

##### 每个创新点针对的问题

1、针对样本不均衡问题：a new training method based on leave-one-out cross validation；

2、针对识别目标小且分布散乱：new learner architectures that localize each object of interest before classifying it；（这个框架是否可以使用到目标检测中）

3、针对细粒度识别：straight forward, parameter-free adjustments can significantly improve performance. In particular, we find that the representational power of the learner can be significantly increased by leveraging bilinear pooling。

