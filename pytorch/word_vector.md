## <center>词向量</center>

词向量（word vector）需要解决一个什么问题呢？现实生活中，我们使用语言文字进行交流，但是这些文字在计算机中就是一串的0和1，而且对于不同的编码方式，编码长度还不一样。因此，要使用计算机对这些文字进行机器学习处理是很不方便的。这个时候我们就希望将这些词编码为一个格式化的向量。

我们希望这个格式化的向量有下面的一些特性：

> 1、格式化的向量之间的close程度表明了两个词之间的意思的相似程度；
>
> 2、要有一定的推断能力，例如：A - B  = C - D,  A是中国， B是深圳， C是美国， D是加州

本文中将主要介绍skip-gram 

### skip-gram

skip-gram做了什么事呢？就是通过当前词来预测相邻词。与这个相似的还有一种叫做CBOW的方法，通过前面的词来预测当前词。

关于skip-gram可以查看原始的论文 和 这篇[博客](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) ，知乎上这篇[文章](https://zhuanlan.zhihu.com/p/29305464)我觉得是讲的比较好的，解决了我为什么这输入和输出不使用同一个embedding层的疑惑，但是还是没有理解输出的embedding的含义以及这两者的具体关系（**希望知道的劳烦告知一下**），同时我也试过使用同一个embedding发现训练不出来。

### 具体的代码

#### dataset文件

```python
# -*- coding:utf-8 -*-

import torch.utils.data as tud
from collections import Counter
import numpy as np
import torch

K = 100


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, path, MAX_VOCAB_SIZE, half_window_size):
        super(WordEmbeddingDataset, self).__init__()
        self.MAX_VOCAB_SIZE = MAX_VOCAB_SIZE
        self.word_to_idx = {}
        self.word_freqs = torch.Tensor([])
        self.idx_to_word = torch.Tensor([])
        self.text_encoded = torch.LongTensor([])
        self.half_window_size = half_window_size
        self.I = 0

        self.read_words(path)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, item):
        center_word = self.text_encoded[item]
        # 防止item-self.half_window_size小于零
        pos_indices = list(range(item - self.half_window_size, item)) + list(range(item+1, item + self.half_window_size + 1))
        # item + self.half_window_size + 1 可能会大于整个text的长度
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]

        neg_words = torch.multinomial(self.word_freqs, K*pos_words.shape[0], True)

        return center_word, pos_words, neg_words

    def read_words(self, path):
        with open(path, 'r') as fin:
            text = fin.read()

        text = text.split()
        vocab = Counter(text)  # 统计text中每个元素出现的次数，顺序按照元素出现顺序给出
        vocab = vocab.most_common(self.MAX_VOCAB_SIZE - 1)  # 找到出现次数最多的
        vocab = dict(vocab)
        vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))  # 不常见的单次的数量

        self.idx_to_word = [word for word in vocab.keys()]
        self.word_to_idx = {word: i for i, word in enumerate(self.idx_to_word)}

        self.word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
        self.word_freqs = self.word_counts / np.sum(self.word_counts)
        self.word_freqs = self.word_freqs ** (3. / 4.)
        self.word_freqs = self.word_freqs / np.sum(self.word_freqs)

        self.text_encoded = [self.word_to_idx.get(word, self.word_to_idx["<unk>"]) for word in text]  # text单词对应的idx
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_freqs = torch.Tensor(self.word_freqs)

```

**需要注意的地方**：

1、第30行其实起不了注释说的作用；

2、必须要重写\__len__ 和\__getitem__函数

#### model文件

```python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        initrange = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)  # 这个就相当于一个nn.Linear，但是这个内部还做了one hot的转化
        self.in_embed.weight.data.uniform_(-initrange, initrange)  # 对参数进行初始化
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        # input_labels [batch_size]
        # pos_labels [batch_size, (window_size * 2)]
        # neg_labels [batch_size, (window_size * 2 * K)]
        input_embedding = self.in_embed(input_labels)   # batch_size * embed_size
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size,  embed_size] -> [batch_size,  embed_size, 1]
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)
        # [batch_size,  embed_size] -> [batch_size,  window * 2]

        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze(2)
        # [batch_size,  embed_size] -> [batch_size,  window * 2 * K]

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):
        return self.in_emded.weight.data.cpu().numpy()
```

**需要主要的点**：

1、第14行，可以参考[官网](https://pytorch.org/docs/stable/nn.html?highlight=torch%20nn%20embedding#torch.nn.Embedding)，不过也可以按照我注释的理解，就相当于一个Linear层，只不过里面帮我们做了转one hot的操作。

2、第26行，unsqueeze 可以参考[官网](https://pytorch.org/docs/stable/torch.html#torch.unsqueeze)，大致的意思就是在添加一个大小为1的维度，squeeze的操作和这个正好相反

3、第27行，bmm函数可以参考[官网文档](https://pytorch.org/docs/stable/torch.html?highlight=torch%20bmm#torch.bmm)，应该算是三维矩阵乘法。

4、第33行，logsigmoid参考[官网](https://pytorch.org/docs/stable/nn.html#torch.nn.LogSigmoid)，这个做了数值稳定处理。

5、这里给我最大的疑惑的就是第22到24行，问什么要使用两个不同的embedding。

### train文件

```python
# -*- coding:utf-8 -*-

import torch
import random
import numpy as np
import torch.utils.data as tud
from word2vector.dataset import WordEmbeddingDataset
from word2vector.model import EmbeddingModel

USE_CUDA = torch.cuda.is_available()

# set parameters
C = 3  # context window
K = 100 # number of negative samples
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 128
LEARNNING_RATE = 0.2
EMBEDDING_SIZE = 100


def set_seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    if USE_CUDA:
        torch.cuda.manual_seed(num)


def main():
    set_seed(1)
    path = "text8.train.txt"
    dataset = WordEmbeddingDataset(path, MAX_VOCAB_SIZE, C)
    dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if USE_CUDA:
        model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNNING_RATE)

    for e in range(NUM_EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()
            if USE_CUDA:
                input_labels = input_labels.to(device)
                pos_labels = pos_labels.to(device)
                neg_labels = neg_labels.to(device)

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                print("epoch", e, "iteration", i, loss.item())


if __name__ == "__main__":
    main()
```



这里没有验证，因为我觉得重点在数据集准备，网络结构和训练上。



### reference



[Skip-Gram: NLP context words prediction algorithm]( https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c )

[理解 Word2Vec 之 Skip-Gram 模型]( https://zhuanlan.zhihu.com/p/27234078 )



更多关于编程和机器学习资料请关注FlyAI公众号。
![公众号二维码][1]

[1]: http://wshaow.club/wechat/%E5%BE%AE%E4%BF%A1%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.jpg