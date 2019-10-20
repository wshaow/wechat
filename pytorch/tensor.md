## <center>pytorch tensor</center>

`Tensor` 与 `Numpy `中的 `ndarrays` 类似， 但是` pytorch` 中的 `Tensor`可以在`GPU`中进行计算。

### 安装pytorch

#### 确定cuda版本

```powershell
nvcc -V
```

通过对应的 `cuda`版本和`python`版本到`pytorch`官网选择合适的安装命令。

我电脑上的是`cuda8.0 `和 `python3.6 `

#### 安装pytorch和torchvision

`pytorch1.2` 对应的 `torchvision`版本是`0.4` ；` pytorch` `1.0`对应的是 `torchvision0.2`

将对应版本的安装包下载到本地使用：

```powershell
pip install 对应的.whl文件
```

#### 查看是否安装成功

```python
import torch
```

如果`import`成功基本就安装成功了

### torch tensor基本操作

#### 创建未初始化的tensor

```python
import torch

x = torch.empty(5, 3)
print(x)

# 输出
tensor([[1.7753e+28, 7.0744e+31, 7.4392e+28],
        [3.8952e+21, 4.4650e+30, 1.1096e+27],
        [2.7518e+12, 7.5338e+28, 1.8589e+34],
        [7.0374e+22, 1.4353e-19, 2.7530e+12],
        [7.5338e+28, 1.5975e-43, 0.0000e+00]])
```

#### 创建一个随机数tensor

```python
import torch

x = torch.rand(5, 3)
print(x)

# 输出
tensor([[0.2603, 0.2963, 0.0465],
        [0.2261, 0.9824, 0.0727],
        [0.9614, 0.8542, 0.9653],
        [0.3685, 0.0180, 0.7694],
        [0.9625, 0.4338, 0.1446]])
```

#### 创建一个初始化为0， 数据类型为long的tensor

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x, x.dtype)

# output
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]) torch.int64
```

#### 使用现有数据初始化一个tensor

```python
x = torch.tensor([5.5, 3])
print(x, x.dtype)

# output
tensor([5.5000, 3.0000]) torch.float32
```

#### 根据现有张量来创建张量

这些方法将重用输入张量的属性，例如：`dtype`除非设置新值进行覆盖

```python
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3)
y = torch.randn_like(x)

print(x, x.dtype)
print(y, y.dtype)

# output
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]]) torch.float32
tensor([[ 0.0688, -0.7617, -0.6564],
        [-1.8253,  0.2384,  0.2155],
        [-1.8708,  1.5294,  0.2178],
        [-0.7267,  0.4747, -0.0252],
        [-0.6143, -1.3407, -1.1383]]) torch.float32
```

我们基于原始的`x`创建新的`x`和`y` 可以看到在原有基础张量 `x`上创建的张量`x` `y`的`dtype`类型是没有变的

除非重新设置张量的属性，例如：

```python
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)
y = torch.randn_like(x, dtype=torch.float)

print(x, x.dtype)
print(y, y.dtype)

# output
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64) torch.float64
tensor([[-0.4164, -0.9306, -1.0304],
        [-0.6035,  0.1864,  0.7424],
        [ 0.7032,  0.2981,  0.2382],
        [-0.1999, -0.6827, -1.1757],
        [-1.0285, -0.1197, -1.0872]]) torch.float32

# x.new_* 方法用来创建对象
```

#### 获取tensor的size

```python
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)

print(x, x.dtype, x.size())
print(x.shape)
# output
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64) torch.float64 torch.Size([5, 3])
torch.Size([5, 3])
# torch.Size()返回一个tuple，支持所有的tuple操作
```

### tensor的运算操作

#### tensor加法

```python
x = torch.rand(5, 3)
y = torch.randn_like(x)

print("直接使用加法运算符：\n", x + y)
print("使用torch.add函数：\n", torch.add(x, y))

# output
直接使用加法运算符：
 tensor([[ 0.0097, -2.0310,  0.3904],
        [ 2.8270,  0.8106,  1.6285],
        [ 0.7797, -0.1262,  0.2150],
        [-0.8661, -0.4480, -0.4136],
        [ 1.2837,  0.5111,  0.3113]])
使用torch.add函数：
 tensor([[ 0.0097, -2.0310,  0.3904],
        [ 2.8270,  0.8106,  1.6285],
        [ 0.7797, -0.1262,  0.2150],
        [-0.8661, -0.4480, -0.4136],
        [ 1.2837,  0.5111,  0.3113]])
```

`torch.add`函数可以传入一个值，用于保存输出结果到给定的这个值中,这样在内部就会少分配依次内存，上面这两个会将结果保存在一个新的内存中，哪怕你将结果重新赋值给x或者y。其实下面这个就相当于是`y`的`in_place`操作了。

```python
x = torch.rand(5, 3)
y = torch.randn_like(x)

print(x+y)

torch.add(x, y, out=y)
print(y)

# output
tensor([[ 2.2646,  0.5464,  1.4707],
        [ 0.5485, -0.3850,  2.0071],
        [-0.4700,  1.4534,  0.7404],
        [ 1.1943,  0.0803,  0.5207],
        [ 0.7237,  1.5434,  0.7937]])
tensor([[ 2.2646,  0.5464,  1.4707],
        [ 0.5485, -0.3850,  2.0071],
        [-0.4700,  1.4534,  0.7404],
        [ 1.1943,  0.0803,  0.5207],
        [ 0.7237,  1.5434,  0.7937]])
```



#### in_place操作

```python
x = torch.rand(5, 3)
y = torch.randn_like(x)

print(x+y)

y.add_(x)
print(y)

# output
tensor([[ 1.2216,  0.6527,  2.0144],
        [-0.4289,  1.4325, -0.9516],
        [ 1.1779,  0.5441,  0.1874],
        [ 0.9043, -0.7059,  2.3566],
        [-1.9184, -0.1799, -1.3181]])
tensor([[ 1.2216,  0.6527,  2.0144],
        [-0.4289,  1.4325, -0.9516],
        [ 1.1779,  0.5441,  0.1874],
        [ 0.9043, -0.7059,  2.3566],
        [-1.9184, -0.1799, -1.3181]])

```

`pytorch`中所有的以**下划线结尾的函数**都是`in_place`操作

#### 分片操作

```python
x = torch.rand(5, 3)
print(x)
print(x[1:, 1:])

# output
tensor([[0.4398, 0.7354, 0.8704],
        [0.4064, 0.2043, 0.2950],
        [0.2368, 0.4293, 0.3025],
        [0.3734, 0.3105, 0.4829],
        [0.1450, 0.5629, 0.2828]])
tensor([[0.2043, 0.2950],
        [0.4293, 0.3025],
        [0.3105, 0.4829],
        [0.5629, 0.2828]])
```

####  Resizeing

可以使用`torch.view`, 如果某个维度的你不想算可以使用-1代替，函数内部会自动帮你计算好这个维度的数据个数，但是前提是要能得到一个整数。view会创建一个新的张量！！！

```python
x = torch.rand(4, 4)
x = x.view(2, 8)
print(x.size())
x = x.view(2, -1)
print(x.size())
x = x.view(-1, 2)
print(x.size())

# output
torch.Size([2, 8])
torch.Size([2, 8])
torch.Size([8, 2])
```

### Numpy 和 Tensor之间的转换

#### tensor转numpy

```python
x = torch.rand(4, 4)
print(x)
y = x.numpy()
print(y)

# output
tensor([[0.6051, 0.3427, 0.3721, 0.5684],
        [0.3118, 0.4058, 0.8179, 0.4685],
        [0.6066, 0.3263, 0.1143, 0.8941],
        [0.0900, 0.1118, 0.5070, 0.9394]])

[[0.60505605 0.34271568 0.3720982  0.5684481 ]
 [0.31178665 0.40575826 0.81789374 0.4685197 ]
 [0.60657334 0.32631427 0.11428601 0.8940574 ]
 [0.09004259 0.11177826 0.507017   0.93936133]]
```

如果tensor在gpu上要先将其搬回到cpu才能转换为numpy。

**上面这两个是共享一个内存空间的，只要一个改变另一个也会随之改变**

#### numpy转tensor

```python
x = np.ones(5)
print("ori_x: ", x)

y = torch.from_numpy(x)
y = y + 1  # 这个时候x不会变化的，因为对y重新分配了内存
print("after y + 1, x: ", x)
print("after y + 1, y: ", y)

z = torch.from_numpy(x)  # 这种情况会改变x的值
torch.add(z, 1, out=z)
print("after torch.add(z, 1, out=z), x: ", x)
print("after torch.add(z, 1, out=z), z: ", z)

# output
ori_x:  [1. 1. 1. 1. 1.]
after y + 1, x:  [1. 1. 1. 1. 1.]
after y + 1, y:  tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    
after torch.add(z, 1, out=z), x:  [2. 2. 2. 2. 2.]
after torch.add(z, 1, out=z), z:  tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

### CUDA Tensor

#### 查看当前机器上是否有GPU可用

```python
print(torch.cuda.is_available())

# output
True
```

#### 获取当前机器CUDA设备对象

```python
device = torch.device("cuda")
print(device)

# output
cuda
```

#### 将张量创建在GPU上

```python
x = torch.randn(5, 3)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    print(y)
    
# output
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
```

#### 将张量移动到GPU上

```python
x = torch.randn(5, 3)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(x)

    x = x.to(device)  # 这里会创建一个新的在GPU上的张量
    print(x)
```

这里也可以将张量移动到CPU上，将device的参数改为cpu即可。





更多关于编程和机器学习资料请关注FlyAI公众号。
![公众号二维码][1]

[1]: http://wshaow.club/wechat/%E5%BE%AE%E4%BF%A1%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.jpg

