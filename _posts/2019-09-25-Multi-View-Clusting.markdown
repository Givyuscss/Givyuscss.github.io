---
layout:     post
title:      "Spectral Clustering"
subtitle:   " \"Spectral Clustering 原理以及简易实现\""
date:       2019-09-25 12:00:00
author:     "Givyuscss"
header-img: "img/post-bg-2015.jpg"
tags:
    - Machine Learning
---


## Spectral Clutering ———— 谱聚类
+ **谱聚类**:谱聚类是利用拉普拉斯算子和图边缘来表示数据点之间的相似度，以此来解决最小切(min-cut)问题的技术。和其他常用的方法相比，谱聚类可以应用于任意形状的聚类，而k-means等仅适用于球形数据的聚类。

  step1 给出一个无向图$$G=(V,E)$$，且向量组$$V=v_1,…,v_N$$，该图的数据邻接矩阵定义为$W$，其中每个$$w_{ij}$$表示$$v_i,v_j$$两个向量之间的相似度。如果$$w_{ij}=0$$意味着$$v_i,v_j$$之间无连接。显然W是一个对称矩阵。

  step2 定义度矩阵$$D$$，为一个对角矩阵，对角元素为$$d_1,….,d_N$$，其中$$d_i$$为$$W$$中第i行数据之和$$d_i=\sum^N_{j=1}w_{ij}$$，则拉普拉斯算子为$$D-W$$，标准化拉普拉斯算子为$$\tilde{L}=D^{-1/2}(D-W)D^{-1/2}$$，在很多谱聚类的应用中，$$L=I-\tilde{L}$$被用来将最小化问题转化为最大化问题。因此$$L,\tilde{L}$$都被成为标准化的拉普拉斯算子。

  因此单视角的谱聚类方法可以表示为：

  $$
  \begin{cases} &\mathop{max}\limits_{U\in \mathbb{R}^{N\times K}}tr(U^TLU)\\ &s.t \quad U^TU=I \end{cases}
  $$

  可以等价转化为：

  $$
  \begin{cases} &\mathop{min}\limits_{U\in \mathbb{R}^{N\times K}}tr(U^T\tilde{L}U)\\ &s.t. \quad U^TU=I\end{cases}
  $$

  $$U$$矩阵的行为数据点的嵌入，可以直接交给k-means来获得最终的聚类结果。解决上述优化问题的方法为：选择矩阵$$U$$，将$$L或\tilde{L}$$从小到大排列取前K个特征值的特征向量作为矩阵$$U$$的列，并对$U$每行特征向量正规化后进行聚类。

**谱聚类Python实现代码：**

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

聚类效果对比：

Spectral clustering:

<img src="/img/in-post/clustering/result_sp.png" width="400px" height="275px"/>

Kmeans：

<img src="/img/in-post/clustering/result_km.png" width="400px" height="275px"/>
