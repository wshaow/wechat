## Numpy实现两层神经网络

### 模型和参数设置

下图为两层神经网络的基本结构和参数的形式，这里输入`X`的特征是按照列向量的形式组织的

![两层神经网络模型和参数](http://wshaow.club/pytorch/two_layer_nn_model.jpg-water_mark)

### 前向传播和反向传播

这里激活函数选择的是`relu`函数，其实关键是反向传播过程求解参数`W1`和`W2`的梯度。如果知道右边的求导图应该还是比较容易写出来的，接下来可能就是什么地方需要加转置（方法就是按照我们求的梯度的大小来确定）。

![前项传播和反向传播推导](http://wshaow.club/pytorch/twolayernnforward-backward.jpg-water_mark)

### numpy实现两层神经网络

```python
import numpy as np

if __name__ == '__main__':
    # forward pass
    N, D_in, H, D_out = 64, 1000, 100, 10  # data num， data in, hidden layer cell num, output num
    # 随机创建一些训练数据
    X = np.random.randn(D_in, N)
    Y = np.random.randn(D_out, N)

    W1 = np.random.randn(D_in, H)
    W2 = np.random.randn(H, D_out)

    learning_rate = 1e-6
    for it in range(500):
        # forward pass
        h = W1.T.dot(X)  # N * H
        h_relu = np.maximum(h, 0)  # N * H
        y_pred = W2.T.dot(h_relu)  # N * D_out

        # compute loss
        loss = np.square(y_pred - Y).sum()
        print(it, loss)

        # backward pass
        # compute the gradient
        grad_y_pred = 2.0 * (y_pred - Y)
        grad_w2 = h_relu.dot(grad_y_pred.T)
        grad_h_relu = W2.dot(grad_y_pred)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = X.dot(grad_h.T)

        # update weights of w1 and w2
        W1 -= learning_rate * grad_w1
        W2 -= learning_rate * grad_w2
```





更多关于编程和机器学习资料请关注FlyAI公众号。
![公众号二维码][1]

[1]: http://wshaow.club/wechat/%E5%BE%AE%E4%BF%A1%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.jpg

