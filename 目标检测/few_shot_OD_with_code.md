## <center> few shot object  detection with code</center>

### [One-Shot Instance Segmentation](https://arxiv.org/abs/1811.11507)

author: Claudio Michaelis 德国 图宾根大学 

#### Task  setup

Given a scene image and a previously unknown object category defined by a single reference instance, generate a bounding box and a segmentation mask for every instance of that category in the image.

Replace the widely used category-based object detection task by an example-based task setup.之前是给定一系列固定的类别，然后检测出测试图像对应的这些类别的物体。现在变成了，给定一个类别的特写样本，只找出这个类别的物体即可。

#### Contribution

- We introduce one-shot instance segmentation, a novel one-shot task, requiring object detection
and instance segmentation based on a single visual example.
- We present Siamese Mask R-CNN, a system capable of performing one-shot instance
segmentation.
- We establish an evaluation protocol for the task and evaluate our model on MS-COCO.
- We show that, for our model, targeting the detection towards the reference category is the
main challenge, while segmenting the correctly identified objects works well.

#### Challenges

在训练集中可能存在我们要检测的物体，但是在训练时并没有把标注它。

> the model will see objects from the test categories during training, but is never provided with any information about them.

### [One-Shot Instance Segmentation](https://arxiv.org/abs/1811.11507v1)

##### 基本框架

![](http://wshaow.club/paper/osis_diffmaskrnn_siamese.PNG)

上图中，IF指的是image feature。其他部分熟悉faster RCNN都比较好理解，关键就是右边框架图的`Feature matching`处理的部分。还有一个比较明显的区别就是：对于maskRCNN，分类结果是给出对应的类别，但是现在只需要给出是不是匹配的的变成了一个二分类问题。



##### Feature matching

![](http://wshaow.club/paper/osis_featurematching.PNG)

主要的步骤：

> 1. Average pool the features of the reference image to an embedding vector. In the fewshot case (more than one reference) compute the average of the reference features as in prototypical networks.
> 2. Compute the absolute difference between the reference embedding and that of the scene at each (x,y) position.
> 3. Concatenate this difference to the scene representation.
> 4. Reduce the number of features with a 1 x 1 convolution.

相比于普通的mask RCNN增加了reference image 和 scene的 feature match 信息。

#### 实验结果

![](http://wshaow.club/paper/osis_results_show.PNG)

但是客观指标并不好，而且对于场景比较杂的情况出错率比较高。



#### Difference between semantic segmentation （语义分割） and  instance segmentation（实例分割）

语义分割是对所有的物体的每一个像素进行分类（会有一个问题，下面的这个图说明语义分割只能区分类别，不能区分个体）， 实例分割分类的是目标检测中对应目标的像素进行高亮，实际上就是把目标检测的bounding box 变成了一个mask，背景这些不处理。

![语义分割](https://pic2.zhimg.com/80/v2-a0e8be79238485e6867f23caeeb97825_hd.jpg)



![实例分割](https://pic4.zhimg.com/v2-dbb56a65bcb6c7eedfd833445fdf9ecf_r.jpg)



### [Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector](https://arxiv.org/pdf/1908.01998v1.pdf)

作者：Qi Fan 腾讯

#### Contributions

> **First**, we propose a general few-shot detection model that can be applied to detect novel objects without re-training and fine tuning. Our method fully exploits the matching relationship between object pairs in a siamese network at multiple network stages. Experiments show that our model can benefit from the attention module at an early stage which enhances the proposal quality, and from the multi-relation module at a late stage that suppresses and filter out false detection on confusing backgrounds.
>
> **Second**, to train the model, we build a large well-annotated dataset with 1000 categories and a few examples for each category. This dataset promotes the general learning of object detection.

这里面对后续比较有用的是提供的数据集。数据集见下面：

![]( http://wshaow.club/FSOD_ARPN.PNG )



#### 基本框架

![]( http://wshaow.club/FSODT-archi.PNG )

这篇文章写得不怎么好，很多地方都没怎么分析，包括怎么就算是引入了Attention，后面三个分支如何实现的他所说的作用。对应的Loss函数也没有说明。



### [Meta-learning algorithms for Few-Shot Computer Vision](https://arxiv.org/pdf/1909.13579v1.pdf)

这是一篇实习报告，很适合meta learning入门。开头介绍了很多few shot 基本的研究现况和常见的一些方法，然后重点放在了meta learning。

主要内容包括

> 1. an extensive review of the state-of-the-art in few-shot computer vision;
> 2. a benchmark of meta-learning algorithms for few-shot image classification;
> 3. the introduction to a novel meta-learning algorithm for few-shot object detection, which is still in development.

这篇文章后面作者尝试使用meta Learning 和 YOLO做一个框架，但是没有完成。



### [Few-shot object detection via feature reweighting](https://arxiv.org/pdf/1812.01866v2.pdf)

ICCV2019 应该是2019年唯一的一篇发表在顶会上的少样本目标检测文章

#### Contributions

> 1. We study the problem of few-shot object detection, which is of great practical values but a less explored task than image classification in the few-shot learning literature. 
> 2. We design a meta-learning based model, where the feature importance in the detector are predicted by a meta-model based on the object category to detect. The model is simple and highly data-efficient. 
> 3. We demonstrate through experiments that our model can outperform baseline methods by a large margin, especially when the number of labels is extremely low.
> 4. Through ablation studies we analyze the effects of certain components of our proposed model and reveal some good practices towards solving the few-shot detection problem.

这里文章有一个观点，在base数据集中出现的数据中存在与新加入的数据集中比较相似的，所以在检测时可以对这些base中数据集的特征可以有一个attention考量。比如：base中有很多猫和飞机的样本，但是现在少量样本数据中加入了狗。显然相对于飞机，猫在预训练中肯定存在更大的意义。

 论文的核心思想，简言之：某些基础类的中间特征，可以适用于其他类的识别。 

主要框架：

![框架](http://wshaow.club/paper/fsod_fr_arc.PNG-water_mark)

图片中如果同一目标有多个不能处理。

[Few-shot adaptive faster R-CNN](https://arxiv.org/pdf/1806.04728v3.pdf)



### reference

[超像素、语义分割、实例分割、全景分割 傻傻分不清？](https://zhuanlan.zhihu.com/p/50996404)