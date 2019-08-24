---

title: 2019-8-17 包含min函数的栈

tags: 算法,每日一题,栈

---



## 包含min函数的栈



### 1. 问题描述

定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。



### 2. 题目解析

这里的很关键的一点是时间复杂度为O(1)。这说明肯定是要有个东西能够随时记录下当前栈中元素的最小值，并且在数据压入或者是弹出栈时也能够随时更新最小值。

#### 2.1 解题思路
这道题有两种解题方法，但是本质其实是一样的：就是使用两个栈，一个栈用于存压入的数据，另一个栈用于存储当前栈的最小值，并且能够随着数据的压入和弹出动态的更新当前栈中的最小值。
**第一种方式**
使用两个栈，数据栈data_stack和最小元素栈min_stack，min_stack栈顶是当前数据栈对应的最小值。
*入栈规则*：对于要压入的数据value，先将这个数压入数据栈data_stack中。如过这个数value<=min_stack.top(),则将这个数压入min_stack中，否则压入min_stack.top()对应的值到min_stack。
*出栈规则*：从min_stack中弹出一个数据，从data_stack中弹出数据作为函数的返回结果。
*min函数* ：返回min_stack.top()对应的值即可。

**第二种方式**
使用两个栈，数据栈data_stack和最小元素栈min_stack，min_stack栈顶是当前数据栈对应的最小值。
*入栈规则*：对于要压入的数据value，先将这个数压入数据栈data_stack中。如过这个数value<=min_stack.top(),则将这个数压入min_stack中。
*出栈规则*：从如果数据栈栈顶的数小于等于min_stack栈顶的数，则从min_stack中中弹出一个数据；从data_stack中弹出数据作为函数的返回结果。
*min函数* ：返回min_stack.top()对应的值即可。

可以看出第二种方式要节省空间，这里只给出第二种方式的实现。
``` C++
class Solution {
private:
    stack<int> data_stack;
    stack<int> min_stack;
public:
    void push(int value) {
        data_stack.push(value);
        if(min_stack.empty()) min_stack.push(value);
        if(value <= min_stack.top()) min_stack.push(value);
    }
    void pop() {
        if(min_stack.empty()) return;
        if(data_stack.top() == min_stack.top()){
            data_stack.pop();
            min_stack.pop();
            return;
        }
        data_stack.pop();
    }
    int top() {
        return data_stack.top();
    }
    int min() {
        return min_stack.top();
    }
};
```

更多关于编程和机器学习资料请关注FlyAI公众号。
![公众号二维码][1]

  [1]: http://pwfic6399.bkt.clouddn.com/wechat/%E5%85%AC%E4%BC%97%E5%8F%B7%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.jpg


