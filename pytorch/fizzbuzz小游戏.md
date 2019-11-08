## <center>FizzBuzz游戏</center>

FizzBuzz游戏规则：从1开始数数，当遇到3的倍数时，说Fizz，当遇到5的倍数时说Buzz，当遇到15的倍数时说FizzBuzz。

### 通过规则实现

直接上代码

```python
def fizz_buzz_encode(num):
    if num % 15 == 0:
        return 3
    elif num % 5 == 0:
        return 2
    elif num % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(num, code):
    return [str(num), 'Fizz', 'Buzz', 'FizzBuzz'][code]


def helper(num):
    print(fizz_buzz_decode(num, fizz_buzz_encode(num)))
    
if __name__ == "__main__":
    helper(10)
```

### 通过神经网络实现

先上代码通过代码来进行说明

```python
# -*- coding:utf-8 -*-
import numpy as np
import torch


NUM_DIGITS = 10


def fizz_buzz_encode(num):
    if num % 15 == 0:
        return 3
    elif num % 5 == 0:
        return 2
    elif num % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(num, code):
    return [str(num), 'Fizz', 'Buzz', 'FizzBuzz'][code]


def helper(num):
    print(fizz_buzz_decode(num, fizz_buzz_encode(num)))


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])  # [::-1]将向量前后翻转


if __name__ == '__main__':
    # for i in range(1, 100):
    #     helper(i)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)]).to(device)
    trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)]).to(device)

    NUM_HIDDEN = 100
    model = torch.nn.Sequential(
        torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(NUM_HIDDEN, 4)
    )

    if torch.cuda.is_available():
        model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    BATCH_SIZE = 128
    for epoch in range(10000):
        for start in range(0, len(trY), BATCH_SIZE):
            end = start + BATCH_SIZE
            batchX = trX[start:end]
            batchY = trY[start:end]

            y_pred = model(batchX)
            loss = loss_fn(y_pred, batchY)

            print('Epoch', epoch, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)]).to(device)

    with torch.no_grad():
        testY = model(testX)

    gtY = np.array([fizz_buzz_encode(i) for i in range(1, 101)])

    print(device, "accuracy:", np.where(gtY == testY.max(1)[1].cpu().numpy())[0].shape[0] / gtY.shape[0])

    predictions = zip(range(1, 101), testY.max(1)[1].cpu().numpy())
    print([fizz_buzz_decode(i, x) for i, x in predictions])

```

**说明**

1、第28行， binary_encode函数做了一个简单的特征工程，将所有的数转换为二进制。

2、这里只是设定了二进制的位数为10位。这样的话相当于总共就只能表示0到$2^{10}$ 这些数，使用101到$2^{10}$用作训练数据，1到100作为测试数据。产生训练数据的函数看第37行。

3、第41行， 构建模型，两层全连接神经网络隐含层神经元个数为100。这里输出为4，是因为我们采用`one-hot`的形式来表示最终的结果，也就是现在结果总共有4种，假如结果为0，那么`one-hot`的形式为`1 0 0 0`以此类推。

4、第50行， 定义了loss函数为[交叉熵loss](https://zhuanlan.zhihu.com/p/38241764)。

5、第71行，在进行测试的时候要使用`with torch.no_grad():`。

6、第76行， 这里的`max`函数参数表示对那个维度求最大值(0对应列， 1对应行)，这里是1，因此会返回每一行的最大值，放在第一维tensor中，第二维tensor会给出最大值在这一行中的位置。

7、第47、48行要放在51行之前。

8、最终的结果：在cpu上的训练效果要比在cuda上的好，cpu的准确率可以达到百分之97， cuda只有百分之91。



更多关于编程和机器学习资料请关注FlyAI公众号。
![公众号二维码][1]

[1]: http://wshaow.club/wechat/%E5%BE%AE%E4%BF%A1%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.jpg