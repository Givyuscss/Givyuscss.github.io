---
layout:     post
title:      "Bayes's basis"
subtitle:   " \"Bayes 基本概念及应用\""
date:       2019-09-29 12:00:00
author:     "Givyuscss"
header-img: "img/post-bg-js-module.jpg"
tags:
    - Machine Learning
---

### 贝叶斯公式
---

+ **贝叶斯法则** 
	
通常，事件A在事件B发生的条件下的发生的概率，与事件B在事件A发生的条件下发生的概率是不一样的。贝叶斯法则就是来陈述这两者之间确定关系的。

+ **贝叶斯公式**

贝叶斯将其思想总结为一条公式：

$$
P(A_i\vert B)=\frac{P(A_i)P(B\vert A_i)}{\sum^n_{i=1}P(A_i)P(B\vert A_i)}
$$

其中各个部分被称为：

$$P(A),P(B)$$被称为事件$$A$$，$$B$$的先验概率或是边缘概率。

$$P(A\vert B)$$被称为事件B发生后A事件发生的条件概率，同理$$P(B\vert A)$$为事件$$A$$发生后$$B$$事件发生的条件概率

贝叶斯公式的推导十分简单，即从条件概率公式推出:

$$
P(A\vert B)=\frac{P(A\cap B)}{P(B)},P(B\vert A)=\frac{P(B\cap A)}{P(A)}
$$

则合并两个式子即可得到：

$$
P(A\vert B)P(B)=P(A\cap B)=P(B\vert A)P(A)
$$

&nbsp;
### 贝叶斯公式用于分类
---
+ **实例**

假设商店来的顾客有$$\{x_1,x_2,...,x_n\}$$个特征，最终需要对顾客进行预测的即为其会不会在商店内购买商品。

通过已有的样本可以计算出针对购买情况$$Y_i$$的各个特征的后验概率：

$$
P(x_1\vert Y_i),P(x_2\vert Y_i),...,P(x_n\vert Y_i)
$$

当新样本携带特征出现时，可以根据以上后验概率和贝叶斯公式来预测样本的购买情况：

$$
P(Y_i\vert \{x_1,x_2,...,x_n\})=\frac{P(Y_i)P(\{x_1,x_2,...,x_n\}\vert Y_i)}{P(X)}
$$

从已有标注的样本中可以获得各个特征的先验概率，且由于假设各个特征相互独立，公式中的$$P(\{x_1,x_2,...,x_n\}\vert Y_i)=\prod^n_{j=1} P(x_j\vert Y_i)$$


+ **文本分类**

文本分类是通过将文本中的句子分割成一个个词汇，通过将词嵌入进向量中作为特征。通过各个词的出现与否来判断该文本属于哪个类别。

首先导入数据，数据分为`trian`和`test`两个文件夹，`trian`中包含`travel`和`hotal`两个类别的txt。
<img src="/img/in-post/Bayes/1.png" width="800px" height="600px"/>

通过分词函数包`jieba`将读取的txt文件内容分成词。
<img src="/img/in-post/Bayes/2.png" width="800px" height="600px"/>

从`sklearn`中导入词向量嵌入函数并将训练样本嵌入。
<img src="/img/in-post/Bayes/3.png" width="800px" height="400px"/>

将样本分为训练集和验证集进行训练兵输出测试结果。
<img src="/img/in-post/Bayes/4.png" width="800px" height="600px"/>

将训练后的模型测试测试数据并输出结果。
<img src="/img/in-post/Bayes/5.png" width="800px" height="600px"/>
<img src="/img/in-post/Bayes/6.png" width="800px" height="400px"/>

[数据与代码](https://github.com/Givyuscss/Givyuscss.github.io/tree/master/code/bayes_datasets "code and data"). 