---
layout:     post
title:      "Spectral Clustering"
subtitle:   " \"Spectral Clustering 原理以及简易实现\""
date:       2019-09-25 12:00:00
author:     "Givyuscss"
header-img: "img/post-bg-js-module.jpg"
tags:
    - Machine Learning
---


## Spectral Clustering

+ 在看《A Survey on Multi-View Clustering》时，文中有提到，谱聚类是多视角聚类的重要基础算法，因此学习记录一下谱聚类的原理和算法实现。
+ 与Kmeans等“传统算法”相比，谱聚类有更好的性能且实现简单。

---

### 1.应用场景

给定一组数据点$$\{x_1,x_2,...,x_n\}$$，以及数据点之间的相似度$$s_{ij}$$，表示$$x_i$$和$$x_j$$数据点之间的相似度。将所有数据点分为K类。使得类内相似度高，类间相似度低。

### 2.算法工具

+ **邻接矩阵$$W$$**：构建关于向量$$V=\{v_1,v_2,...,v_n\}$$的无向图$$G(V,E)$$，$$W$$为$$G$$的邻接矩阵，其中的$$w_{ij}$$表示$$v_i,v_j$$之间的连接权重。当$$w_{ij}=0$$时，表示两个向量无连接，且显然$$w_{ij}=w_{ji}$$。邻接矩阵$$W$$通过相似度矩阵$$S$$得到，有三种常见的方法：

  **1)$$\epsilon$$-邻近**：根据相似度矩阵$$S$$中的$$s_{ij}=\|x_i-x_j\|^2$$

  $$
  w_{ij}=
  \begin{cases}
  0,&s_{ij}>\epsilon\\
  \epsilon,&s{ij}\leq \epsilon
  \end{cases}
  $$

  **2)K邻近**:K邻近有两种方法，第一种是当两个向量同时在对方的K邻近中才满足，第二种是有一>个向量在另一个向量的K邻近中即可，此时的$$w_{ij}$$均为:$$w_{ij}=w_{ji}=e^{\frac{\|x_i-x_j\|^2}{2\sigma^2}}$$,反之为0.

  **3)全连接（高斯）**:$$w_{ij}=w_{ji}=e^{\frac{\|x_i-x_j\|^2}{2\sigma^2}}$$

+ **度矩阵$$D$$**：另设定关于向量族的度矩阵$$D$$，$$d_{ij}=\sum^n_{j=1}w_{ij},w_{ij}\in W,d_{ij}\in D$$。度矩阵被定义为对角元素为$$[d_1,d_2,...,d_n]$$的对角矩阵。

+ **拉普拉斯矩阵$$LO$**：定义为正则化的拉普拉斯矩阵为$$L=D-W$$
+ $$L$$有如下性质：
  1)$$\forall f\in R^n,\grave{f}Lf=\frac{1}{2}\sum^n_{i,j=1}w_{ij}(f_i-f_j)^2$$

  2)对称，半正定

  3)最小特征值为0，对应常数特征项量1

  4)有n个非负特征值，且 $$ 0= \lambda_1 \leq \lambda_2 \leq ... \leq \lambda_n $$

+ 有两种正规化的方式：
  1)随机游走：$$L_{rw}=D^{-1}L=1-D^{-1}W$$
  2)对称：$$L_{sym}=D^{-1/2}WD^{-1/2}$$

### 3.未正规化谱聚类算法步骤

1. 通过相似矩阵$$S$$建立邻接矩阵$$W$$，设定分类个数k
2. 通过邻接矩阵计算度矩阵$$D$$
3. 计算拉普拉斯矩阵$$L$$
4. 计算$$L$$的前k小的特征值所对应的特征向量$$\{u_1,u_2,...,u_k\}$$
5. 将$$\{u_1,u_2,...,u_k\}$$中每个向量作为矩阵$$U$$的列
6. 设$$y_i$$为$$U$$中第i行的向量，即$$y_1=[u_{11},u_{12},...,u_{1k}]$$
7. 采用Kmeans对$$y_i,i=1,2,..,n$$分类，分出结果$$A_1,A_2,...,A_k$$

**Python实现：**

```python
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

#生成数据点，make_moons生成为两个月牙形
def circle_data(num_sample=500):
    X,y= datasets.make_moons(n_samples=num_sample,noise=0.08)
    return X,y

#生成邻接矩阵W
def w_matrix(sample_data):
    length = len(sample_data)
    W = np.zeros((length,length))
    dis_matrix = np.zeros((length,length))
    #计算距离矩阵D
    for i in range(length):
        for j in range(i+1,length):
            dis_matrix[i][j] = np.linalg.norm(sample_data[i] - sample_data[j])
            dis_matrix[j][i] = dis_matrix[i][j]
    #通过KNN生成邻接矩阵W
    for idx,each in enumerate(dis_matrix):
        index_array  = np.argsort(each)
        W[idx][index_array[1:10+1]] = 1  # 距离最短的是自己
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2
    return W

#生成度矩阵D
def d_matrix(Wmatrix):
    length = len(Wmatrix)
    d = np.zeros((length,length))
    for i in range(length):
        d[i][i]=np.sum(Wmatrix[i])
    return d

#生成随机颜色标记不同类别的数据点
def randRGB():
    return (random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0)

#根据数据点标签生成图像
def plot(matrix,C,n_clustering):
    colors = []
    for i in range(n_clustering):
        colors.append(randRGB())
    for idx,value in enumerate(C):
        plt.scatter(matrix[idx][0],matrix[idx][1],color=colors[int(C[idx])])
    plt.show()

#预设参数
num_sample = 500
n_clustering = 2

#生成W，D，L矩阵
X,_ = circle_data(num_sample)
W = w_matrix(X)
D = d_matrix(W)
L = D - W

#计算拉普拉斯矩阵的特征值和特征向量
x,V = np.linalg.eig(L)
dim = len(x)
dictEigval = dict(zip(x,range(0,dim)))
#排序并选取前K个特征值所对应的特征向量
kEig = np.sort(x)[0:n_clustering]
ix = [dictEigval[k] for k in kEig]
x,V = x[ix],V[:,ix]

#用KMeans对特征向量聚类
sp_cluster = KMeans(n_clusters=n_clustering).fit(V)
plot(X,sp_cluster.labels_,n_clustering=n_clustering)
```

### 聚类效果对比：

+ **Spectral clustering:**

<img src="/img/in-post/clustering/result_sp.png" width="400px" height="275px"/>

+ **Kmeans：**

<img src="/img/in-post/clustering/result_km.png" width="400px" height="275px"/>
