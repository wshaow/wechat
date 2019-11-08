## <center>语言模型</center>

语言模型，就是用一个模型来描述语言。在上世纪主要七八十年代以前，主要有两种方式：一种是基于语法结构和规则的描述方式，另一种是基于统计的。但是随着数据量和计算机算力的提升，基于统计的效果碾压基于语法结构和规则的方法。语言模型的历史发展和一些主要的技术可以去看吴军老师的《数学之美》。

基于统计的语言模型主要做的事就是一句话出现的概率。这里要先看一下概率论的链式法则：

![链式法则](http://wshaow.club/pyotrch/language_model/language-model-chainrule.jpg-water_mark)

这个存在一个比较大的问题，就是当前词出现的概率与之前所有出现的词都有关，这样的话计算量将十分大。于是乎就有了markov假设：

![markov](http://wshaow.club/pyotrch/language_model/language-model-markov.jpg-water_mark)

语言模型的评价

 语言模型**复杂度**（**Perplexity**）：在给定上下文的条件上，句子中每个位置平均可以选择的单词数量。**复杂度**越小，每个位置的词就越确定，模型越好。 

![evaluate](http://wshaow.club/pytorch/language_model/language-model-evaluate.jpg-water_mark)

### 常见的语言模型

#### 循环神经网络（Recurrent neural network， RNN）

![RNN](http://wshaow.club/pytorch/language_model/language-model-rnn.jpg-water_mark)

训练RNN比较困难，

