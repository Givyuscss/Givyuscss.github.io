---
layout:     post
title:      "Multi-View-Clusting"
subtitle:   " \"Multi-View-Clusting 详解\""
date:       2019-09-25 12:00:00
author:     "Givyuscss"
header-img: "img/post-bg-2015.jpg"
tags:
    - 生活
---



# A Survey on Multi-View Clustering

## I Abstract

### 1.Definition:

MVC是一个机器学习方法：通过结合多视角信息将相似的个体聚类，将不同的物体分开。

### 2.categories:

+ **generative (or model-based) approaches ———生成（基于模型）方法**

尝试学习数据的基本分布，并用生成模型来表示数据分类情况，一个模型代表一个类别

+ **discriminative (or similarity-based) approaches ——— 区分（基于相似性）方法**

直接优化一个与相似性相关的目标函数，使得类别内的个体相似度增加，不同类别之间的相似性减小

### 3.Discriminative apporaches

区分方法应用广泛，基于他们不同的组合多视角信息方法将其分为5类：

+ **common eigenvector matrix 公共特征向量矩阵(主要为多视角光谱聚类)**

- **common coefficient matrix 公共系数矩阵(主要为多视角子空间聚类)**

- **common indicator matrix 公共指标矩阵(主要为多视图非负矩阵因子聚类)**

- **direct view combination 直接视图组合(主要为多内核集群)**

- **view combination after projection 投影后视图组合(主要为典型相关分析(CCA)**

前三个方法的相似之处是他们使用了相近的结构来结合多视角信息

## II Gerenative approaches

大多数情况，生成方法是基于混合模型和EM算法的

### **1. Mixture Models 混合模型**

混合分布可以被表示为：
\begin{equation}
\displaystyle p(x|\theta)=\sum^{K}_{k=1}\pi_kp(x|\theta)
\end{equation}

+ **以混合高斯模型(GMM)为例：**

**单高斯模型（正态分布）**:$\displaystyle f(x)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\gamma)^2}{2\sigma^2})$，$\gamma$和$\sigma^2$分别表示高斯分布的均值和方差

如下图所示服从二维高斯模型的数据会聚集在一个椭圆内，服从三维高斯模型的数据会聚集在一个椭球内

<img src="/img/in-post/clustering/2dGs.png" width="400px" height="275px"/>

**混合高斯模型：**如下图所示，数据并不是由一个模型生成，因此单个高斯模型无法代表数据的特征。通过求解两个高斯模型，并通过一定权重将两个高斯模型融合成一个，即最终的高斯混合模型。

<img src="/img/in-post/clustering/mixgs.png"  width="400" height="275"/>

假设混合高斯模型由K个模型混合而成，则GMM的概率密度函数为：
$$
\displaystyle p(x)=\sum^{K}_{k=1}p(k)p(x|k)=\sum^{k}_{k=1}\pi_kN(x|u_k,\sum_{k})
$$
其中$p(x|k)=N(x|u_k,\sum_{k})$表示第K个模型的概率密度函数，$\pi_k$表示第K个模型所占权重，$\sum^{K}_{k=1}\pi_k=1$

+ **最大期望算法（Expectation-maximization algorithm,EM）**

令**X**表示已观测数变量集，**Z**表示隐变量集，$\Theta$表示参数模型，需要对$\Theta$做极大似然估计，则需要最大化似然函数：

$$LL(\Theta|X,Z)=lnP(X,Z|\Theta)$$

算法主要依赖于无法观测的隐变量，基本思想是：若参数$\Theta$已知，则可根据训练数据推断出最优隐变量**Z**的值(E step);反之若Z的值已知，则对参数$\Theta$做极大似然估计(M step)

以初始值$\Theta^0$为起点，执行以下步骤直至收敛：

1. 基于$\Theta^t$推断隐变量**Z**的期望，记作$Z^t$
2. 基于已观测变量**X**和$Z^t$对参数$\Theta$做极大似然估计，记作$\Theta^{t+1}$

+ **CMMs**

  给出一个数据集$X=x_1,x_2,…,x_N\in\mathbb{R}^{d\times{N}}$，则混合模型CMM分布为:$ Q(x)=\sum^{N}_{j=1}q_jf_j(x),x\in\mathbb{R}^{d\times{N}}$

  且$q_j\ge1$表示第j个集合的先验概率，满足$\sum^{N}_{j=1}q_j=1$。$f_j(x)$是一个指数族分布，它的期望等于第j个数据点。

  考虑质数函数和布雷格曼散度(Bergman divergences)的双射关系，指数族分布$f_j(x)=C_\phi(x)exp(\beta d_\phi(x,x_j))$中，$d_\phi$表示布雷格曼散度，$C_\phi$与$x_j$独立不相关，$\beta$控制分布的情况。

  因此CMMs的目的是将指数似然函数:

  $$
\begin{split} L(X;\{q_j\}^N_{j=1})&=\frac{1}{N}\sum^N_{i=1}log(\sum^N_{j=1}q_jf_j(x_i))\\&=\frac{1}{N}\sum^N_{i=1}log(\sum^N_{j=1}q_je^{-\beta d_\phi(x_i,x_j)})+const \end{split}
  $$
对数似然函数可以等价表示为$\hat{P}$和$Q(x)$的KL距离(相对熵):
  

相对熵：$\displaystyle D(P|Q)=\sum_{x\in X}P(x)log\frac{P(x)}{Q(x)}$
$$
\begin{split} \displaystyle D(\hat{P}|Q)&=\sum^N_{i=1}\hat{P}(x_i)log\frac{\hat{P}(x_i)}{Q(x_i)}\\&=\sum^N_{i=1}\hat{P}(x_i)[log\hat{P}(x_i)-logQ(x_i)]\\&=-\sum^{N}_{i=1}\hat{P}(x_i)logQ(x_i)-\mathbb{H}(\hat{P})\\&=-L(X;\{q_j\}^N_{j=1})+const \end{split}
$$
其中$\mathbb{H}(\hat{P})$表示P的信息熵，与参数$q_j$相独立。

接下来问题转化为最小化目标，可以通过迭代算法解决，因此先验概率迭代公式为：

$$
  \displaystyle q^{(t+1)}_j=q^{(t)}_j\sum^N_{i=1}\frac{\hat{P}f_j(x_i)}{\sum^N_{j`=1}q^{(t)}_{j`}f_{j`}(x_i)}
$$
  迭代后，根据K个$q_j$值最大的点作为参照，将剩余的点分配给与之有最高先验概率的参照点，将数据点分成K个不相交的集。

  **发现**聚类的效果受$\beta$所影响，所以初始值$\beta_0$的选取通常采用$\beta_0=N^2logN/\sum^N_{i,j=1}d_\phi(x_i,x_j)$来保证$\beta$能在一个合理的范围内。

### **2. 基于混合模型或EM算法的多视角聚类**

对于多视角的CMMs，每个$x_i$都有m个视角，即$x_i=\{x^1_i,x^2_i,…,x^m_i\}$则每个视角的混合分布为$Q^v(x^v)=\sum^N_{j=1}q_jf^v_j(x^v)=C_\phi(x^v)\sum^N_{j=1}q_je^{-\beta^vd_{\phi_v}(x^v,x^v_j)}$

为了取得一个结合所有视角的公共聚类结果，所有的$Q^v(x^v)$共享同样的先验权重。则目标函数为：

$$\begin{split} \mathop{min}\limits_{q_1,…,q_N}&\sum^m_{v=1}D(\hat{P}^v|Q^v) \\&=\mathop{min}\limits_{q_1,…,q_N}\{-\sum^m_{v=1}\sum^N_{i=1}\hat{P}^v(x^v_i)logQ^v(x^v_i)-\sum^m_{v=1}\mathbb{H}(\hat{P}^v)\} \end{split}$$

可以看出优化目标是凸的，存在全局最优解。迭代公式为：

$$\displaystyle q^{(t+1)}=\frac{q^{(t)}_j}{M}\sum^m_{v=1}\sum^N_{i=1}\frac{\hat{P}f^v_j(x^v_i)}{\sum^N_{j`=1}q^{(t)}_{j`}f^v_{j`}f^v_{j`}(x^v_i)}$$

$q_j$表示第j个数据作为范例的置信度

## III Discriminative Approaches

+ **区别**：和生成方法不同，区分方法直接优化目标函数来寻找最优的聚类方案，而不是现根据样本建立模型再依据模型来确定聚类结果。
+ 至今为止，大多数的MVC方法都属于区别方法，目标是将N个物体聚类成K个类别，最终获得一个隶属矩阵$H\in\mathbb{R}^{N\times K}$来表述分类情况，可见矩阵中每行的和为1。当每行只有一个元素为1其他均为0时，成为硬聚类，否则为软聚类。

#### **1.公共特征向量矩阵**

该类方法主要是基于十分常用的谱聚类技术，通过假定所有视角都拥有相同或者相似的特征向量矩阵，来获得聚类结果。

**主要有两个代表性的方法**：1.co-training spectral clustering,2.co-regularized spectral clustering.

+ **谱聚类**:谱聚类是利用拉普拉斯算子和图边缘来表示数据点之间的相似度，以此来解决最小切(min-cut)问题的技术。和其他常用的方法相比，谱聚类可以应用于任意形状的聚类，而k-means等仅适用于球形数据的聚类。

  step1 给出一个无向图$G=(V,E)$，且向量组$V=v_1,…,v_N$，该图的数据邻接矩阵定义为$W$，其中每个$w_{ij}$表示$v_i,v_j$两个向量之间的相似度。如果$w_{ij}=0$意味着$v_i,v_j$之间无连接。显然W是一个对称矩阵。

  step2 定义度矩阵$D$，为一个对角矩阵，对角元素为$d_1,….,d_N$，其中$d_i$为$W$中第i行数据之和$d_i=\sum^N_{j=1}w_{ij}$，则拉普拉斯算子为$D-W$，标准化拉普拉斯算子为$\tilde{L}=D^{-1/2}(D-W)D^{-1/2}$，在很多谱聚类的应用中，$L=I-\tilde{L}$被用来将最小化问题转化为最大化问题。因此$L,\tilde{L}$都被成为标准化的拉普拉斯算子。

  因此单视角的谱聚类方法可以表示为：

  $$\begin{cases} &\mathop{max}\limits_{U\in \mathbb{R}^{N\times K}}tr(U^TLU)\\ &s.t \quad U^TU=I \end{cases}$$

  可以等价转化为：

  $$\begin{cases} &\mathop{min}\limits_{U\in \mathbb{R}^{N\times K}}tr(U^T\tilde{L}U)\\ &s.t. \quad U^TU=I\end{cases}$$

  $U$矩阵的行为数据点的嵌入，可以直接交给k-means来获得最终的聚类结果。解决上述优化问题的方法为：选择矩阵$U$，将$L或\tilde{L}$从小到大排列取前K个特征值的特征向量作为矩阵$U$的列，并对$U$每行特征向量正规化后进行聚类。

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

+ **co-training multi-view spectral clustering**

  对于半监督学习，当同时提供标注和无标注的数据时，两个视角协同训练是被广泛认可的方法。该方法假定基于两个视角所建立的模型有很高的几率能够给出相同的分类结果。

  在多视角谱聚类协同训练中，拉普拉斯矩阵的特征向量中包含着聚类的区别信息，因此谱聚类多视角协同训练将特征向量在一个视角中进行聚类，并将结果用于改善另一个视角中的拉普拉斯矩阵。

  邻接矩阵$W_{N\times N}$可以被看作是一个N维的向量，表示第i个点和其他所有点的相似度。因为最大的K个特征向量包含着聚类的分类信息，因此可以沿着这几个放心投影相似向量，来保留聚类细节信息，丢弃会混淆分类的群体内部细节。

+ **Co-Regularized Multi-View Spectral Clustering:**

  Co-regularized（共正则化）对于半监督多视角学习是一个十分有效的技术。它的主要思想是减小两个视角预测函数之间的差异使得其整合为一个目标函数。

  共正则化谱聚类采用帕普拉斯矩阵的特征向量来作为半监督学习中的预测方程。

  令$U^{(s)},U^{(t)}$作为拉普拉斯矩阵$L^{(s)},L^{(t)}$对应的特征向量。

  1. 第一个版本采用成对的共正则化标准来使得$U^{(s)},U^{(t)}$越接近越好，用$D$来衡量两个视角的分类分歧程度：

  $$\displaystyle D(U^{(s)},U^{(t)})=\|\frac{K^s}{\|K^{(s)}\|^2_F}-\frac{K^{(t)}}{\|K^{(t)}\|^2_F}\|^2_F$$

  $$K^{(s)}=U^{(s)}{U^{(s)}}^T$$使用的线性核是$U^{(s)}$的相似矩阵。当$\|K^{(s)}\|^2_F=k$时，k是聚类的数量，两个视角的聚类分歧可以被表示为:

  $$\displaystyle D(U^{(s)},U^{(t)})=-tr(U^{(s)}{U^{(s)}}^TU^{(t)}{U^{(t)}}^T)$$

  将所有的差异方程整合进一个目标函数中，则共正则化谱聚类可以被写作如下优化问题：

  $$\begin{cases} &\mathop{max}\limits_{U^{(1)},U^{(2)},…,U^{(m)}}\sum^m_{s=1} ({U^{(s)}}^TL^{(s)}U^{(s)}) +\sum_{1\le s,t\le m,s\ne t}\lambda tr(U^{(s)}{U^{(s)}}^TU^{(t)}{U^{(t)}}^T) \\ &s.t.{U^{(s)}}^TU^{(s)}=I,\forall 1\le s\le m. \end{cases}$$

  超参数$\lambda$用来协调谱聚类目标和谱聚类嵌入分歧的权重。取得嵌入结果之后，每个$U^s$均可以输入给kmeans进行聚类，最终的结果会有些微的不同。

  2. 第二个版本是centroid-based co-regularization，

#### **2.公共系数矩阵**

很多现实中的例子，即使给出的数据维度很高，但是真正能够代表数据特征的维度其实很低。就像给出的图像维度很高，但是描述图像颜色形状的参数并不多。在实际应用中，数据可以被分成多个子空间，子空间聚类是一种寻找底层子空间，并根据已经辨别的子空间来对数据点进行聚类的技术。

+ **子空间聚类**：子空间聚类利用了数据的自表达性，每个样本可以用其他几个数据样本的线性组合表示，经典的子空间聚类表达是：$X=XZ+E$。其中$Z={z_1,z_2,…,z_N}\in \mathbb{R}^{N\times N}$是子空间系数矩阵，每一个$z_i$是原本数据点$x_i$对于子空间的表示。$E\in \mathbb{R}^{N\times N}$是噪声矩阵。

  子空间聚类可以被写作如下优化问题:

  $\begin{cases}&\mathop{min}\limits_{Z}\|X-XZ\|^2_F \\&s.t. \ Z(i,i)=0,Z^T1=1 \end{cases}$

  其中约束$Z(i,i)=0$是为了防止数据点由自身表示，而约束$Z^T1=1$表示数据点位于仿射子空间的并集中。$Z_i$中的非零元素和相同子空间中的数据点相符。

  得到了子空间表示$Z$后，可以获得相似矩阵$W=\frac{|Z|+\Z^T|}{2}$来构建拉普拉斯算子，最后在拉普拉斯矩阵上运行谱聚类即可得到最终的聚类结果。

+ **多视角子空间聚类**：每个视角都可以得到一个子空间表示$Z_v$，为了从多视角获得统一的聚类结果，通过使得每对视角的系数矩阵差异减小，来得到公共系数矩阵。优化问题可以被写作:

  $$\begin{cases} &\mathop{min}\limits_{Z^{(s)},s=1,2,…,m}\sum^m_{s=1}\|X^{(s)}-X^{(s)}Z^{(s)}\|^2_F+\alpha\sum^m_{s=1}\|Z^{(s)}\|_1+\beta\sum_{1\le s\le t}\|Z^{(s)}-Z^{(t)}\|_1 \\&s.t. \ diag(Z^{(s)})=1, \forall s\in \{1,2,…,m\} \end{cases}$$

  其中$\|Z^{(s)}-Z^{(t)}\|_1$是基于成对公正则化约束的L1范数，可以减少噪声问题的影响。

#### **3.公共指示矩阵**

+ **非负矩阵分解**（NMF）：对于一个非负数据矩阵$X\in \mathbb{R}^{d\times N}_+$，非负矩阵分解寻找两个非负矩阵因子$U\in \mathbb{R}^{d\times K}_+,V\in \mathbb{R}^{N\times K}_+$，使得它们的乘积是X的近似：$X \approx UV^T$，其中K是希望聚类的种数，$U$是基础矩阵，$V$指示矩阵。

+ **基于NMF的多视角聚类**:在非负矩阵分解框架中结合多视角信息，通过NMF中不同视角的公共指示矩阵来进行多视角聚类。然而指示矩阵可能无法在相同的尺度上进行比较。为了使得聚类方案在不同的视角都有用意义且可以比较，提出了一个新的约束，将每个视角的指示矩阵整合成一个公共指示矩阵，受到NMF和概率隐性语义分析之间的关系的启发，引出了另一个归一化约束。最终的优化问题的结构为：

  $$\begin{cases}&\mathop{min}\limits_{U^{(v)},V^{(v)},v=1,2,…,m}\sum^m_{v=1}\|X^{(v)}-U^{(v)}V^{(v)}\|^2_F+\sum^m_{v=1}\lambda_v\|V^{(v)}-V^*\|^2_F\\&s.t.\ \forall 1\le k\le K,\|U^{(v)}_{.,k}\|_1=1,U^{(v)},V^{(v)},V^{(*)}\ge 0\end{cases}$$

+ **多视角KMeans**:Kmeans聚类可以借助NMF来引入一个指示矩阵$H$。则基于NMF的Kmeans聚类的公式为:

  $$\begin{cases} &\mathop{min}\limits_{H,G}\|X^T-HG^T\|^2_F \\&s.t. \ H_{i,k}\in \{0,1\},\sum^K_{k=1}H_{i,j}=1,\forall i=1,2,…,N \end{cases}$$

  因为kmeans算法不需要耗费很大的计算资源，它主要基于特征分解，所以对于大尺度的数据聚类来说，它是一个很好地选择。为了解决大尺度多视角数据，提出了一个通过采用不同视角的公共指示矩阵进行多视角kmeans聚类的方法，优化问题的结构为：

  $$\begin{cases} &\mathop{min}\limits_{G^{(v)},\alpha^{(v)},H}\sum^m_{v=1}(\alpha^{(v)})^\gamma\|{X^{(v)}}^T-HG^T\|_{2,1}\\&s.t.\ H_{i,k}\in \{0,1\},\sum^K_{i,k}=1,\sum^m_{v=1}\alpha^{(v)}=1 \end{cases}$$

  其中$\alpha^{(v)}$是第v个视角的权重，$\gamma$是控制权重的参数。通过学习来获得不同视角的权重，重要的视角的权重将会增加。

#### **4.直接结合法**

除了这些通过共享不同视角中的数据结构，通过核直接结合视角信息也是一个十分常用的多视角聚类方法。很自然的方法是对每个视角都定义一个核，再将这些核结合成一个凸组合。

+ **核方程和核组合方法**：核是一个仅用线性学习算法来解决非线性问题的策略，因为核方程$K:\mathcal{X} \times \mathcal{X} \to \mathbb{R}$可以直接给出特征空间中的内积，而不用定义非线性变换$\phi$，如下是常用的核方程:

  1. 线性核:$K(x_i,x_j)=(x_i\cdot x_j)$
  2. 多项式核：$K(x_i,x_j)=(x_i\cdot x_j+1)^d$
  3. 高斯核:$K(x_i,x_j)=exp(-\frac{\|x_i-x_j\|^2}{2\sigma^2})$
  4. Sigmoid核：$K(x_i\cdot x_j)=tanh(\eta x_i\cdot x_j+v )$

  在生成核希尔伯特空间中，核方程可以被看作是向量空间中的相似方程，所以采用核作为谱聚类和核kmeans方法中的非欧几里得相似度量。

  如果将每个视角中的核组合起来去解决多视角问题，这就是多视角聚类的多核学习方法。显然，多核学习被认为是该类多视角聚类方法中最重要的一种。主要有如下三种组合多核的方法：

  1. 线性组合：分为线性求和和加权线性求和两个子类。
  2. 非线性组合
  3. 基于数据的组合

+ **核kmeans核谱聚类**：使得$\phi(\cdot):x\in \mathcal{X} \to \mathcal{H}$为一个特征映射，将$x$映射为生成希尔伯特空间的元素$\mathcal{H}$，则核kmeans方法写作如下形式：

  $$\begin{cases} &\mathop{min}\limits_{H}\sum^N_{i=1}\sum^K_{k=1}H_{ik}\|\phi(x_i)-\mu_k\|^2_2\\&s.t. \ \sum^K_{k-1}H_{ik}=1 \end{cases}$$

#### **5.投影结合法**

​		对于有着相同结构的多视角数据，直接结合他们是很方便的事。但是在现实应用中，多视角的数据可能会有不同的结构，难以直接进行比较和结合。例如，病人的生物基因信息和临床症状作为两个聚类分析的视角，这难以直接结合。而且，高纬度和噪声难以处理。为了解决这类问题，投影结合法被提出，其中最常用的为CCA和KCCA。

+ **CCA和KCCA**:

  给树两个数据集$S_x=[x_1,x_2,…,x_N]\in\mathbb{R}^{d_x\times N},S_y=[y_1,y_2,…,y_N]\in \mathbb{R}^{d_y\times N}$每个x和y的均值均为0，CCA为x寻找一个投影$w_x\in\mathbb{R}^{d_x}$为y寻找一个投影$w_y\in\mathbb{R}^{d_y}$使得$S_x,S_y$的在$w_x,w_y$上投影的相关系数最大。

  

  其中$\rho$表示相关系数，$C_{xy}$表示x,y的协方差矩阵，均值为0。发现$\rho$不受$w_x,w_y$的影响，CCA可以写作如下形式：

  $$\begin{cases}\mathop{max}\limits_{w_x,w_y}  \ &{w_x}^TC_{xy}w_y\\s.t &{w_x}^TC_{xx}w_x=1\\&{w_y}^TC_{yy}w_y=1\end{cases}$$