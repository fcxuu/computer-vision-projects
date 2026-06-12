# MNIST MLP + CNN（NNDL 课程 PJ1）

**来源仓库**：`cxk049/NNDL`（已归档）

## 项目简介

神经网络与深度学习课程 PJ1 —— MNIST 手写数字分类，**从零搭建神经网络框架**（仅用 NumPy，不依赖 PyTorch/TensorFlow 等自动微分框架）。

实现了 MLP、CNN、各种优化器（SGD、MomentGD）、学习率调度器（StepLR、MultiStepLR、ExponentialLR）、损失函数、多种正则化方法（L2、Early Stopping）和数据增强。

## 文件结构

```
nndl-mnist-mlp-cnn/
├── codes/
│   ├── mynn/                       # 自实现的神经网络库
│   │   ├── op.py                   # 基础算子（Linear, Conv2D, ReLU, Sigmoid）
│   │   ├── optimizer.py            # 优化器（SGD, MomentGD）
│   │   ├── lr_scheduler.py         # 学习率调度器
│   │   ├── models.py               # 模型定义（MLP, CNN）
│   │   ├── runner.py               # 训练流程
│   │   ├── metric.py               # 评价指标
│   │   ├── data_augmentation.py    # 数据增强
│   │   └── visualization.py        # 可视化
│   ├── test_train.py               # MLP 训练脚本
│   ├── test_model.py               # MLP 测试脚本
│   ├── weight_visualization.py     # 权重可视化
│   ├── hyperparameter_search.py    # 超参数搜索
│   └── dataset_explore.ipynb       # 数据集探索
├── HOMEWORK_ORIGINAL.md            # 原 README（保留原文）
└── README.md                       # 本文件
```

## 使用说明

### 训练 MLP 模型

```bash
cd codes
python test_train.py
```

### 测试 MLP 模型

```bash
cd codes
python test_model.py
```

## 实验结果

- **MLP**（784 → 600 → 10，ReLU，MomentGD）：训练准确率 ~98%，测试准确率 ~93%
- **CNN**（1 → 16 → 32 channels，5×5 kernels）：训练准确率 ~99%，测试准确率 ~98%

## 任务描述（原文）

详见 [`HOMEWORK_ORIGINAL.md`](./HOMEWORK_ORIGINAL.md)，即原仓库 README。
