## pytorch实现两层神经网络

具体的模型参数设置在numpy实现两层神经网络一文中已经做了说明，在该文中也对前项传播和反向传播公式进行了推导。

在本文中将只讲述pytorch中对两层神经网络的实现。

### 自己求解反向传播梯度

这个实现可以完全参考numpy的实现，具体代码如下：

```python
import torch

if __name__ == '__main__':
    # forward pass
    N, D_in, H, D_out = 64, 1000, 100, 10  # data num， data in, hidden layer cell num, output num
    # 随机创建一些训练数据
    X = torch.randn(D_in, N)
    Y = torch.randn(D_out, N)

    W1 = torch.randn(D_in, H)
    W2 = torch.randn(H, D_out)

    learning_rate = 1e-6
    for it in range(500):
        # forward pass
        h = W1.t().mm(X)  # N * H
        h_relu = h.clamp(min=0)  # N * H
        y_pred = W2.t().mm(h_relu)  # N * D_out

        # compute loss
        loss = (y_pred - Y).pow(2).sum().item()
        print(it, loss)

        # backward pass
        # compute the gradient
        grad_y_pred = 2.0 * (y_pred - Y)
        grad_w2 = h_relu.mm(grad_y_pred.t())
        grad_h_relu = W2.mm(grad_y_pred)
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = X.mm(grad_h.t())

        # update weights of w1 and w2
        W1 -= learning_rate * grad_w1
        W2 -= learning_rate * grad_w2
```

**这里主要是做了这样几个修改**：

1、numpy中点乘运算是`dot`函数， pytorch中变成了`mm`函数；

2、numpy中用于求relu的函数时`maxnum`函数， pytorch中的[clamp](https://pytorch.org/docs/stable/torch.html#torch.clamp)函数

### 使用pytorch中的自动求导

#### 初识自动求导

```python
import torch

if __name__ == '__main__':
    x = torch.tensor(1., requires_grad=True)
    w = torch.tensor(2., requires_grad=True)
    b = torch.tensor(3., requires_grad=True)

    y = w*x + b

    y.backward()

    print(x.grad)
    print(w.grad)
    print(b.grad)
    print(y.grad)
    
    # output
    tensor(2.)
    tensor(1.)
    tensor(1.)
    None
```

其实只要看过之前反向传播的推导的时候的那个求导图的话是比较容易理解的。说白了就是链式求导法则，就是一层一层的求导，当前位置的导数是上一层的得到的导数与上层位置对当前位置导数的一个乘积，所以每一层都会保存当前位置对应的从求导开始点到现在这个位置的导数值。这个值就是每个`tensor`中的`grad`.

#### 两层网络的自动求导版本

```python
# -*- coding:utf-8 -*-

import torch

if __name__ == '__main__':
    # forward pass
    N, D_in, H, D_out = 64, 1000, 100, 10  # data num， data in, hidden layer cell num, output num
    # 随机创建一些训练数据
    X = torch.randn(D_in, N)
    Y = torch.randn(D_out, N)

    W1 = torch.randn(D_in, H, requires_grad=True)
    W2 = torch.randn(H, D_out, requires_grad=True)

    learning_rate = 1e-6
    for it in range(500):
        # forward pass
        h = W1.t().mm(X)  # N * H
        h_relu = h.clamp(min=0)  # N * H
        y_pred = W2.t().mm(h_relu)  # N * D_out

        # compute loss
        loss = (y_pred - Y).pow(2).sum()
        print(it, loss.item())

        loss.backward()
        # backward pass
        # compute the gradient

        print(W1.grad)  #
        # update weights of w1 and w2
        with torch.no_grad():  # 把这步排除在计算图之外
            W1 -= learning_rate * W1.grad
            W2 -= learning_rate * W2.grad
            W1.grad.zero_()
            W2.grad.zero_()
```

**这里有几个需要注意的地方：**

1、如果要对某个`tensor`求导需要传入参数requires_grad=True

2、tensor的grad应该是每次都会叠加到原有的grad上，因此在每次反向传播前完都需要对grad进行清零（第一次除外）。

3、如果不需要将某一个步骤添加到求导的计算图中需要使用`with torch.no_grad():`包起来。上面如果没有使用这个语法回报以下错误：

```python
RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
```

### 使用torch.nn实现

对于`torch.nn`的具体细节可以参看[官方文档](https://pytorch.org/docs/stable/nn.html?highlight=torch%20nn#module-torch.nn)，内容有点多可以先大致浏览一遍，以后需要用的时候再查文档。

这里要使用到`torch.nn.Sequential`是一个序列容器，关于如何使用和初始化可以查看[官方文档](https://pytorch.org/docs/stable/nn.html#sequential)

然后使用全连接层[torch.nn.linear](https://pytorch.org/docs/stable/nn.html?highlight=torch nn linear#torch.nn.Linear)来构建两层神经网络模型。

```python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

if __name__ == '__main__':
    # forward pass
    N, D_in, H, D_out = 64, 1000, 100, 10  # data num， data in, hidden layer cell num, output num
    # 随机创建一些训练数据
    X = torch.randn(N, D_in)
    Y = torch.randn(N, D_out)

    model = nn.Sequential(
       nn.Linear(D_in, H),
       nn.ReLU(),
       nn.Linear(H, D_out)
    )

    loss_fn = nn.MSELoss(reduction="sum")
    learning_rate = 1e-3

    print(model[0].weight)  # 获取第一层
    for it in range(500):
        # forward pass
        y_pred = model(X)  # model.forward()

        # compute loss
        loss = loss_fn(y_pred, Y)
        print(it, loss.item())
        # zero grad
        model.zero_grad()
        # backward
        loss.backward()

        # update weights
        with torch.no_grad():  # 把这步排除在计算图之外
            for param in model.parameters():  # param{tensor, grad}
                param -= learning_rate * param.grad

```

**上面的代码需要关注**

1、第13行，如何使用Sequential构建模型；

2、第19行，使用nn自定义的[loss](https://pytorch.org/docs/stable/nn.html?highlight=torch%20nn%20mseloss#torch.nn.MSELoss)；

3、第31行，每次在更新导数前要对模型的梯度进行清零；

4、第37行如何获取模型中的参数；

5、第22行，如何获取模型中的某一层和其中的参数。其实如果使用字典的方式初始化Sequential的话，可以使用参数名来获取模型的某一层；

6、可以使用`torch.nn.init`对模型的参数进行初始化，参看[官方文档](https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.normal_)。

### 使用torch.optim进行参数更新

这里要使用到`torch.optim`，可以参看[官方文档](https://pytorch.org/docs/stable/optim.html?highlight=torch%20optim#module-torch.optim)。

具体代码：

```python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

if __name__ == '__main__':
    # forward pass
    N, D_in, H, D_out = 64, 1000, 100, 10  # data num， data in, hidden layer cell num, output num
    # 随机创建一些训练数据
    X = torch.randn(N, D_in)
    Y = torch.randn(N, D_out)

    # create model
    model = nn.Sequential(
       nn.Linear(D_in, H),
       nn.ReLU(),
       nn.Linear(H, D_out)
    )
    # create loss function
    loss_fn = nn.MSELoss(reduction="sum")
    # create optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(model[0].weight)  # 获取第一层
    for it in range(500):
        # forward pass
        y_pred = model(X)  # model.forward()

        # compute loss
        loss = loss_fn(y_pred, Y)
        print(it, loss.item())
        # zero grad
        optimizer.zero_grad()
        # backward
        loss.backward()

        # update model parameters
        optimizer.step()
```

**这里主要需要关注**：

1、第23行，如何创建一个optimizer；

2、第34行，如何将参数梯度置零；

3、第39行，如何更新参数。

### 使用torch.nn.Module

这里主要是如何使用`nn.Module`，当然最好的资料还是[官方文档](https://pytorch.org/docs/stable/nn.html?highlight=torch%20nn%20module#torch.nn.Module)。

具体代码：

```python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        h = self.linear1(x)
        h_relu = self.relu(h)
        y_pred = self.linear2(h_relu)  
        # y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred


if __name__ == '__main__':
    # forward pass
    N, D_in, H, D_out = 64, 1000, 100, 10  # data num， data in, hidden layer cell num, output num
    # 随机创建一些训练数据
    X = torch.randn(N, D_in)
    Y = torch.randn(N, D_out)

    # create model
    model = TwoLayerNet(D_in, H, D_out)
    # create loss function
    loss_fn = nn.MSELoss(reduction="sum")
    # create optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for it in range(500):
        # forward pass
        y_pred = model(X)  # model.forward()
        # compute loss
        loss = loss_fn(y_pred, Y)
        print(it, loss.item())
        # zero grad
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update model parameters
        optimizer.step()


```

**这里主要关注：**

1、第7到第19行，如何利用`nn.Module`构建自己的模型；

2、第11行，对于`nn.Relu`这个是一个类，要先初始化，之后才可以像第16行这么用，不能直接`nn.Relu(x)`这样使用。



更多关于编程和机器学习资料请关注FlyAI公众号。
![公众号二维码][1]

[1]: http://wshaow.club/wechat/%E5%BE%AE%E4%BF%A1%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.jpg