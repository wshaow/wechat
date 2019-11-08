# <center>Meta Learning</center>

## What is meta learning

### 基本概念

>  Meta-learning, also known as “learning to learn”, intends to design models that can learn new skills or adapt to new environments rapidly with a few training examples. 
>
> There are three common approaches:
>
>  1) learn an efficient distance metric (metric-based); 
>
> 2) use (recurrent) network with external or internal memory (model-based);
>
>  3) optimize the model parameters explicitly for fast learning (optimization-based). 

~~The idea of Meta learning comes from the fact that humans can learn a new concept or skill with just a few examples. For example, kids can tell cats and birds apart after they just see a few time of  cats and birds.~~ 

现在的深度学习方法普遍都需要大量的样本来训练网络模型，但是人却能在看过几张小鸟小猫的图片之后就能很快的将它们区分开。因此，人们就想设计出和人的学习一样高效快速的网络模型，meta learning应运而生。人们对meta learning提出的要求：

> 1、通过少量样本学习新的概念或者技能；
>
> 2、要有比较好的适应性和泛化能力；

### 常见的meta learning task

- A classifier trained on non-cat images can tell whether a given image contains a cat after seeing a handful of cat pictures.

- A game bot is able to quickly master a new game.

- A mini robot completes the desired task on an uphill surface during test even through it was only trained in a flat surface environment.

- 我比较关注的是few shot learning

  

### 这里面让我比较困惑的是这个是怎么训练的，训练数据分成了很多类，每种的含义是什么？



### 论文

#### [ Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks ]( https://arxiv.org/pdf/1703.03400.pdf )

作者： Chelsea Finn  加州大学伯克利分校

**contributions：** A simple model and task-agnostic algorithm for meta-learning that trains a model’s parameters such that a small number of gradient updates will lead to fast learning on a new task. 

 **Problem Set-Up** ：

## reference

[Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)

[meta learning](https://www.chainnews.com/articles/023205795025.htm)这篇终于讲了我比较在意的训练样本的拆分了

[Model-Agnostic Meta-Learning （MAML）模型介绍及算法详解](https://zhuanlan.zhihu.com/p/57864886) 对MAML以及其训练过程解释的比较详细



