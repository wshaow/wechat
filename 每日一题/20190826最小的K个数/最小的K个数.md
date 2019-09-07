---
title: 2019-8-26 最小的k个数
tags: 算法,每日一题,矩阵打印
---

## <center>最小的k个数</center>

### 1. 问题描述

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

### 2. 题目解析
这道题目如果单单只是要得到正确结果的话还是很简单的，关键是如果要使时间复杂度达到O(n)，那么难度就会大很多。另外还要对输入参数进行判断。如果 `k` 大于数组大小，直接返回空数组。另外还要判断 `k`是不是小于等于0，如果是也直接返回空数组。还有就是输入数组为空的话也直接返回空数组。其他情况正常处理即可。
#### 2.1 具体思路
这里提供了三种方法：

**方法一**：对数组进行排序，取前k个元素即可。

**方法二**：使用大根堆进行求解，大根堆的大小为k，如果大根堆根节点的值大于当前遍历到的input的值将大根堆根节点弹出，压入当前input位置的值。最后输出大根堆中的k个数即可。

**方法三**：第三种方法是最巧妙的方法，能够是时间复杂度达到O(n)。所以重点说一下这种方法。这种方法要借鉴快速排序中的partition思想，partition做的一个事就是选定一个值我们称为哨兵，然后数组中小于等于哨兵值的数我们放在哨兵值的左边，大于哨兵值的数字我们放在其右边。具体的算法如下：

![数组中出现次数过半的数](http://pwfic6399.bkt.clouddn.com/wechat/daily_topic/%E6%9C%80%E5%B0%8F%E7%9A%84k%E4%B8%AA%E6%95%B0.jpg?imageView2/0/q/75|watermark/2/text/d3NoYW93/font/YXJpYWw=/fontsize/400/fill/I0NBQkFDQQ==/dissolve/73/gravity/SouthEast/dx/10/dy/10|imageslim)

后面发现这种方法有点问题对于这个用例 5,4,4,4,4,5,4,4 k=7发现通不过。我现在固定选定的哨兵就是最后一个第一次partition之后的结果为：

```C++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        vector<int> res;
        if(input.size() < k || k <= 0 || input.size() <= 0) return res;//这里要对输入进行判断，否则要发生错误
        if(input.size() == k) return input;
         
        GetLeastNumbers_Solution3(input, k, res);
        return res;
    }
     
    void GetLeastNumbers_Solution1(vector<int> &input, int k, vector<int>& res){//方法一：排序后取前k个数，复杂度O(nlogn)
        sort(input.begin(), input.end());
        for(int i=0; i<k; ++i){
            res.push_back(input[i]);
        }
    }
     
     void GetLeastNumbers_Solution2(vector<int> &input, int k, vector<int>& res){//方法二：使用大根堆的方式,复杂度O(nlogk)
         //以下两种方法都可以
        //使用优先队列实现
        priority_queue<int> p;
        for (int i = 0; i < input.size(); ++i) {
            if (p.size() < k) {
                p.push(input[i]);
            }
            else {
                if (p.top() > input[i]) {
                    p.pop();
                    p.push(input[i]);
                }
            }
        }
 
        while (!p.empty()) {
            res.push_back(p.top());
            p.pop();
        }
        
         //使用堆实现
        /*res = vector<int>(input.begin(), input.begin() + k);
        make_heap(res.begin(), res.end());
 
            for (int i = k; i < input.size(); ++i)
            {
                if (input[i] < res[0])
                {
                    //先pop,然后在容器中删除
                    pop_heap(res.begin(), res.end());//pop_head并不会删除堆顶元素只是将其放在最后
                    res.pop_back();
                    //先在容器中加入，再push
                    res.push_back(input[i]);
                    push_heap(res.begin(), res.end());
                }
            }*/
     }
    
    void swap(vector<int> &input, int a, int b){
        int temp = input[a];
        input[a] = input[b];
        input[b] = temp;
    }
    
    void GetLeastNumbers_Solution3(vector<int> &input, int k, vector<int>& res){//方法三：使用partition的方式
        int len = input.size();
        int begin = 0, end = len-1;
        
        int sentry = -1;
        while(sentry != k){
            int rand_num =  input[end];
            int less = begin; int great = end;
            
            while(less < great){//partition过程
                if(input[less] > input[end]){
                    swap(input, less, --great);
                }else{
                    ++less;
                }
            }
            
            swap(input, great, end);
            sentry = great;
            
            if(sentry < k){//继续往左边走
                begin = sentry;
            }else{//往左边走
                end = sentry-1;
            }
        }
        
        for(int i=0; i<sentry; ++i){
            res.push_back(input[i]);
        }
    }
};

```

更多关于编程和机器学习资料请关注FlyAI公众号。
![公众号二维码][1]

[1]: http://pwfic6399.bkt.clouddn.com/wechat/%E5%BE%AE%E4%BF%A1%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.jpg