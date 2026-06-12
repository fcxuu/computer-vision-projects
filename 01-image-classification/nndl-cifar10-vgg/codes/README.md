代码都在这里

# 神经网络与深度学习项目2

本项目实现了基于CIFAR-10数据集的图像分类任务，研究不同网络结构和批量归一化技术的影响。

## 项目结构

```
codes/
├── train_cifar10.py               # CIFAR-10训练主脚本
├── VGG_BatchNorm/                 # VGG与BatchNorm对比实验
│   ├── VGG_Loss_Landscape.py      # 损失景观可视化
│   ├── data/                      # 数据加载相关代码
│   │   ├── loaders.py             # 数据加载器
│   │   └── __init__.py
│   ├── models/                    # 模型定义
│   │   ├── vgg.py                 # VGG模型及其变种
│   │   └── __init__.py
│   └── utils/                     # 工具函数
│       ├── nn.py                  # 神经网络辅助函数
│       └── __init__.py
└── results/                       # 结果保存目录（运行后生成）
    ├── models/                    # 保存的模型权重
    └── figures/                   # 生成的图表和可视化
```

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- torchvision
- matplotlib
- numpy
- tqdm

## 运行指南

### 1. CIFAR-10分类网络训练

使用`train_cifar10.py`脚本训练不同的网络架构，可以尝试不同的优化器和超参数：

```bash
# 训练标准VGG-A网络
python train_cifar10.py --model vgg_a --optimizer adam --lr 0.001 --epochs 30

# 训练带批量归一化的VGG-A网络
python train_cifar10.py --model vgg_a_bn --optimizer adam --lr 0.001 --epochs 30

# 训练带Dropout的VGG-A网络
python train_cifar10.py --model vgg_a_dropout --optimizer adam --lr 0.001 --epochs 30

# 使用SGD优化器
python train_cifar10.py --model vgg_a --optimizer sgd --lr 0.01 --epochs 30

# 添加L2正则化
python train_cifar10.py --model vgg_a --optimizer adam --lr 0.001 --weight_decay 1e-4 --epochs 30
```

### 2. 批量归一化(Batch Normalization)实验

使用`VGG_BatchNorm/VGG_Loss_Landscape.py`脚本研究批量归一化对训练过程的影响：

```bash
# 运行VGG-A与VGG-A-BatchNorm对比实验和损失景观可视化
python VGG_BatchNorm/VGG_Loss_Landscape.py
```

该脚本会生成:
- 各个模型在不同学习率下的训练曲线
- 损失景观可视化
- 梯度稳定性分析图
- 模型对比分析图

## 结果查看

训练后的结果将保存在`results`目录下:
- `results/models/`: 保存训练好的模型权重
- `results/figures/`: 生成的图表和可视化结果

你可以通过查看这些图表比较不同模型架构和优化策略的效果。 
