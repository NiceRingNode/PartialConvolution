# PartialConvolution-Inpainting

This is a non-official re-implementation of article: [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)[Liu+, arXiv2018].

The official implementation is [here](https://github.com/NVIDIA/partialconv).

# Requirements

Python 3.7.7+

Pytorch 1.7.0+

```shell
python install -r requirements.txt
```

# Work

The works of this re-implementation contains:

- [x] Partial Convolution Layer

- [x] New Mask Datasets

  After training, it is found that the area of the mask will influence the effect of image inpainting, so this re-implementation uses three mask datasets with different areas proportion and three corresponding weights were trained respectively.

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

In windows 10, download the pretrained weights, [Extract code：jw2x](https://pan.baidu.com/s/1P93LDjkaJvnxwkm4LcnCOw ), and clone the repository.

```
git clone https://github.com/NiceRingNode/PartialConvolution.git
```

Then change the working directory in cmd,

```
cd/d PartialConvolution
```

and run test.py using the following commands,

```shell
python test.py
```

or

```shell
python test.py --batch_size 8 --pretrained_root "./weights/checkpoint_mask_lightest_16.8.pth" --dataset "mask_lightest"


python test.py --batch_size 8 --pretrained_root "./weights/checkpoint_mask_light_23.55.pth" --dataset "mask_light"


python test.py --batch_size 8 --pretrained_root "./weights/checkpoint_mask_35.5.pth" --dataset "mask"
```

You can see the inpainting result on **result.png** in the **output folder**.

# Train

## Preprocess

Download the dataset [Places2](http://places2.csail.mit.edu/download.html), and put it in the data folder, the directory is as follows (the example data set here uses places365_standard, you can replace it with other **Places2** dataset)

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

Then generate the mask dataset, the number of masks is 8000 as default.

```
python generate_mask.py
```

Changing to the working directory in cmd, then run `train.py`

```shell
python train.py
```

# Results

Experiments have proved that if there are some small holes in the middle of the covered part, the inpainting effect will be better.

The following shows the training results using three kinds of masks that have different area proportion, from top to bottom:

> mask image，original image, predict image, comp image, mask

mask：**shadow area: 35.5%**

![shadow area: 35.5%](/output/using_35.5.png)



mask_light: **shadow area: 23.55%**

![shadow area: 23.55%](/output/using_23.5.png)



mask_lightest: **shadow area: 16.8%**

![shadow area: 16.8%](/output/using_16.8.png)

# More

Here, I only provide the **python-based** re-implementation of the paper. The libtorch-based re-implementation has been completed, and it is being deployed on the PC as a desktop software using **C++**. Soon the model will be tried to be deployed on **Android**.

# Reference

- https://github.com/NVIDIA/partialconv
- https://github.com/naoto0804/pytorch-inpainting-with-partial-conv

