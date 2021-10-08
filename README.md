# PartialConvolution

这是一个对[Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)[Liu+, arXiv2018]的**非官方**复现

官方实现在[这里](https://github.com/NVIDIA/partialconv)

# Requirements

Python 3.7.7+

Pytorch 1.7.0+

```shell
python install -r requirements.txt
```

# Work

本复现包含的工作有：

- [x] Partial Convolution Layer

- [x] New Mask Datasets

  训练之后发现，mask的面积会影响图像修复的效果，所以本复现使用了三个不同面积的mask dataset，分别训练出三种对应的权重：

  `checkpoint_mask_lightest_16.8.pth`

  `checkpoint_mask_light_23.55.pth`

  `checkpoint_mask_35.5.pth`

  | name          | area ratio | with holes |
  | ------------- | ---------- | ---------- |
  | mask_lightest | 16.8%      | √          |
  | mask_light    | 23.55%     | √          |
  | mask          | 35.5%      | ×          |

- [x] pytorch weights to libtorch weights

- [x] libtorch inference implementation in C++17 (This work will be published as a desktop application.)

# Test

windows环境下，下载预训练权重，[extract code：jw2x](链接：https://pan.baidu.com/s/1P93LDjkaJvnxwkm4LcnCOw )，

```
git clone https://github.com/NiceRingNode/PartialConvolution.git
```

打开cmd，切换到工作目录

```
cd/d PartialConvolution
```

运行test

```shell
python test.py
```

或者

```shell
python test.py --batch_size 8 --pretrained_root "./weights/checkpoint_mask_lightest_16.8.pth" --dataset "mask_lightest"


python test.py --batch_size 8 --pretrained_root "./weights/checkpoint_mask_light_23.55.pth" --dataset "mask_light"


python test.py --batch_size 8 --pretrained_root "./weights/checkpoint_mask_35.5.pth" --dataset "mask"
```

在output文件夹下找到result.png，查看结果

# Train

## Preprocess

下载数据集[Places2](http://places2.csail.mit.edu/download.html)，放到data文件夹下，目录如下所示（这里的示例数据集使用的是places365_standard，可以更换其他的Places2数据集）

```shell
├─data
│  ├─mask
│  ├─mask_light
│  ├─mask_lightest
│  └─places365_standard
│      ├─train
│      └─val
├─output
├─weights
```

之后生成mask数据集，默认是8000张mask

```
python generate_mask.py
```

cmd中切换到工作目录，运行train.py

```shell
python train.py
```

# Results

实验证明，如果遮掩的部分中间有一些小孔，修复的效果会更好一些

下面展示了使用三种不同面积mask的训练结果，从上到下分别是：

mask image，original image, predict image, comp image, mask

mask：**shadow area: 35.5%**

![shadow area: 35.5%](/output/using_35.5.png)



mask_light: **shadow area: 23.55%**

![shadow area: 23.55%](/output/using_23.5.png)



mask_lightest: **shadow area: 16.8%**

![shadow area: 16.8%](/output/using_16.8.png)

# More

这里只给出了基于python对论文的复现，基于libtorch的复现已经完成，正在写PC端的软件（稍后会开源），之后会尝试将模型部署在Android上
