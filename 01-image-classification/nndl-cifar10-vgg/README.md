# CIFAR-10 VGG + BatchNorm（NNDL 课程 PJ2）

**来源仓库**：`cxk049/NNDL_PJ2`（已归档）

## 项目简介

神经网络与深度学习课程 PJ2 —— CIFAR-10 分类任务，研究不同网络结构和批量归一化（BatchNorm）技术的影响。

实现了 VGG-A、VGG-A-BN、VGG-A-Dropout 三种架构，并附带损失景观（Loss Landscape）可视化与梯度稳定性分析。

## 文件结构

```
nndl-cifar10-vgg/
├── codes/
│   ├── train_cifar10.py                          # CIFAR-10 训练主脚本
│   ├── VGG_BatchNorm/
│   │   ├── VGG_Loss_Landscape.py                 # 损失景观可视化
│   │   ├── models/vgg.py                         # VGG 模型定义
│   │   ├── data/                                 # 数据加载（空，仅有 __init__.py）
│   │   └── utils/nn.py                           # 神经网络辅助函数
│   └── README.md                                 # 原 codes 下的 README
├── HOMEWORK_ORIGINAL.md                         # 原任务描述（保留原文）
├── README_SOURCE.md                             # 原仓库主 README 备份
└── README.md                                     # 本文件
```

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- torchvision
- matplotlib
- numpy
- tqdm

## 使用说明

### 1. 训练 CIFAR-10 分类网络

```bash
cd codes

# 训练标准 VGG-A 网络
python train_cifar10.py --model vgg_a --optimizer adam --lr 0.001 --epochs 30

# 训练带 BatchNorm 的 VGG-A
python train_cifar10.py --model vgg_a_bn --optimizer adam --lr 0.001 --epochs 30

# 训练带 Dropout 的 VGG-A
python train_cifar10.py --model vgg_a_dropout --optimizer adam --lr 0.001 --epochs 30

# 使用 SGD 优化器
python train_cifar10.py --model vgg_a --optimizer sgd --lr 0.01 --epochs 30

# 添加 L2 正则化
python train_cifar10.py --model vgg_a --optimizer adam --lr 0.001 --weight_decay 1e-4 --epochs 30
```

### 2. BatchNorm 对比实验 + 损失景观可视化

```bash
cd codes
python VGG_BatchNorm/VGG_Loss_Landscape.py
```

该脚本会生成：
- 各模型在不同学习率下的训练曲线
- 损失景观可视化
- 梯度稳定性分析图
- 模型对比分析图

## 任务描述（原文）

详见 [`HOMEWORK_ORIGINAL.md`](./HOMEWORK_ORIGINAL.md) 和 [`README_SOURCE.md`](./README_SOURCE.md)。
