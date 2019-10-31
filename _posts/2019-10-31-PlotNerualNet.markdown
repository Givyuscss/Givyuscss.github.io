---
layout:     post
title:      "PlotNeuralNet"
subtitle:   " \"绘制神经网络结构工具\""
date:       2019-10-31 00:00:00
author:     "Givyuscss"
header-img: "img/post-bg-js-module.jpg"
tags:
    - 可视化
---


### PlotNeuralNet——绘制神经网络结构工具

+ 组会ppt上需要介绍一个基于VGG的提取声音的网络，找不到现成的网络结构图，只好自己画一个。

+ 无奈在线绘制的结构颜值不够且不能容纳很大的图像，没法保存完整的结构图。如果需要简单直观的可以考虑在线生成的：

  https://cbovar.github.io/ConvNetDraw/

+ 另外用tensorboard导出的图留白太大，不适合展示。

+ 最后找到`PlotNerualNet`这个绘图包，颜值能打，就是可供绘制的层样式太少，绘制简单的图基本够用。

+ 首先从github上下载：

  ```shell
  git clone https://github.com/HarisIqbal88/PlotNeuralNet
  ```

+ `PlotNeuralNet`是通过LaTeX绘图的，需要预装LaTeX环境，由于之前配置过LaTeX环境，这步就省了。

+ 功能介绍写的很简略，参数什么的基本都靠自己摸索了。

+ `PlotNeuralNet`的主要功能是在`pycore`文件夹中，将层转化为LaTeX的内容存在`tikzeng.py`中：

  常用的层大致都有：

  1. to_Conv：卷积层

  2. to_ConvConvRelu：两个卷积层接Relu

  3. To_Pool：池化层

  4. to_SoftMax：softmax分类层

  5. to_connection：连接符号

  6. to_skip：跳转连接

     ....

+ 头疼的事参数的含义，试了半天才搞明白

  以`to_Conv`的参数为例吧：

  ```python
  def to_Conv( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
  ```

  1. `name`很简单，该层的名字，用于和别的层位置关系时使用

  2. `s_file`是显示在每个层的边上，可以代表输入张量的长宽

  3. `n_filer`显示在每个层的下方，可以代表层数，可以按着自己的想法输入合适的数字

  4. `offset`是最头疼的一个数字，看了源代码的发现是和后面的`to`结合使用，表示的是和当前的层`to`代表层的距离:

     ```python
     offset="(0,0,0)",to="(conv1-east)"
     ```

     表示的就是当前层和`conv1`层的距离为0，具体表现就是画出来的图代表两个层的方块没有空隙。`-east`大概就是conv1东边（左边）的意思吧....我猜的。

  5. `Width`、`height`、`depth`三个比较好理解，就是方块的三维尺寸。
  6. `Caption`可以给方块加小标题。

+ 下面是我写的VGGish的结构图：

  ```python
  import sys
  sys.path.append('../')
  from pycore.tikzeng import *
  from pycore.blocks  import *
  
  # defined your arch
  arch = [
      to_head( '..' ),
      to_cor(),
      to_begin(),
      to_input('../examples/download.jpg'),
      to_Conv("conv1",96,64,offset="(0,0,0)", to="(0,0,0)",height=96,depth=64,width=2),
      to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",height=96,depth=64,width=2),
      to_Conv("conv2",48, 128, offset="(1,0,0)", to="(pool1-east)", height=48, depth=32, width=8),
      to_connection( "pool1", "conv2"),
      to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=48, depth=32, width=3),
      to_ConvConvRelu("conv3",24, (256,256), offset="(1,0,0)", to="(pool2-east)", height=24, depth=16, width=(12,12)),
      to_connection( "pool2", "conv3"),
      to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)", height=24, depth=16, width=3),
      to_ConvConvRelu("conv4",12, (512,512), offset="(1,0,0)", to="(pool3-east)", height=12, depth=8, width=(16,16)),
      to_connection( "pool3", "conv4"),
      to_Pool("pool4", offset="(0,0,0)", to="(conv4-east)", height=12, depth=8, width=4),
      to_ConvRes("linear1",1,4096,offset="(1,0,0)", to="(pool4-east)",height=2,depth=2,width=30,opacity=0.9),
      to_connection( "pool4", "linear1"),
      to_ConvRes("linear2",1,4096,offset="(1,0,0)", to="(linear1-east)",height=2,depth=2,width=30,opacity=0.9),
      to_connection( "linear1", "linear2"),
      to_ConvRes("linear3",1,128,offset="(1,0,0)", to="(linear2-east)",height=2,depth=2,width=15,opacity=0.9),
      to_connection( "linear2", "linear3"),
      to_end()
      ]
  
  def main():
      namefile = str(sys.argv[0]).split('.')[0]
      to_generate(arch, namefile + '.tex' )
  
  if __name__ == '__main__':
      main()
  
  ```

+ 还用到了`to_connection`，功能是在两个层之间添加箭头，参数是两个层的名字。

+ `to_input`可以输入示例图像，显示在网络结构的最初位置

+ 可用的层太少了，想给最后全连接层换个颜色，发现没有，只好选了这个`to_ConvRes`代替。

+ 运行命令很简单：

  ```shell
  bash ../tikzmake.sh my_arch
  ```

  一个是`tikzmake.sh`的路径和`my_arch.py`的路径，py文件无需加后缀，因为最后读取的是`tex`文件。

+ 最终效果还是不错的：

  <img src="/Users/givyuscss/Givyuscss.github.io/img/in-post/PlotNerualNet/net.jpg" alt="net" style="zoom:50%;" />