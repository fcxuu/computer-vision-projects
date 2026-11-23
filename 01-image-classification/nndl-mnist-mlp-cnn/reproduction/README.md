# MNIST · 从零实现 MLP 与 CNN

本目录提供新的训练与测试入口。保留课程代码的基本全连接层、激活函数和学习率调度器，引入向量化的 NumPy 卷积、显式 Flatten 反传、L2 只计入权重的一次更新，以及可独立回载的 NPZ。卷积层按 NCHW 接口接收输入，使用窗口展开和矩阵乘法。SciPy 只用于仿射增强中的图像重采样。

## 复现

在本目录执行。脚本从 MNIST 镜像下载原始 IDX 数据并校验 MD5，默认缓存在仓库根目录的 `data/mnist/`。

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
export VECLIB_MAXIMUM_THREADS=4 OPENBLAS_NUM_THREADS=4

python verify_layers.py
python experiment.py train --out runs/full
python experiment.py test --run runs/full
python report.py --run runs/full --out runs/full/report
```

`train` 支持 `--trials` 指定配置、`--patience` 启用早停。亦可用 `test --checkpoint evidence/cnn_affine.npz` 单独评估某次运行保存的权重。

## 实验设计

官方 60,000 张训练图按类分成 50,000 张训练和 10,000 张验证，种子 309。每组统一跑 15 轮，batch 128。增强组以 0.5 的概率对训练图联合施加 ±15° 旋转、±2 像素平移及 0.9–1.1 倍缩放。验证和测试使用原图。

| 配置 | 架构 | 增强 | 初始学习率 | 动量 | 调度器 | 权重 L2 |
|---|---|---|---:|---:|---|---:|
| mlp_plain | MLP | 否 | 0.04 | 0.9 | ExponentialLR | 0.0001 |
| mlp_affine | MLP | 是 | 0.04 | 0.9 | ExponentialLR | 0.0001 |
| mlp_sgd | MLP | 是 | 0.12 | 0 | ExponentialLR | 0.0001 |
| mlp_step | MLP | 是 | 0.02 | 0.9 | StepLR | 0.0001 |
| cnn_plain | CNN | 否 | 0.02 | 0.9 | MultiStepLR | 0.00001 |
| cnn_affine | CNN | 是 | 0.02 | 0.9 | MultiStepLR | 0.00001 |

MLP 为 784 → 600 → 10。CNN 使用 16/32 通道的 5×5 卷积，stride 2、padding 2，然后连接 100 单元隐藏层和 10 类输出。每组按最高验证准确率选权重，并列时比较验证交叉熵，再并列保留较早轮。所有组训练结束后评估官方 10,000 张测试图。训练交叉熵按更新前的批次输出累计，可与干净验证集曲线结合观察。

数据来源：[MNIST 数据集](https://yann.lecun.com/exdb/mnist/) · [Torchvision 官方数据集校验值](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)。

## 实测结果

| 配置 | 最佳验证轮次 | 验证准确率 | 官方测试准确率 |
|---|---:|---:|---:|
| mlp_plain | 15 | 97.72% | 97.96% |
| **mlp_affine** | **15** | **98.43%** | **98.57%** |
| mlp_sgd | 15 | 97.58% | 97.91% |
| mlp_step | 15 | 98.14% | 98.42% |
| cnn_plain | 15 | 98.69% | 98.71% |
| **cnn_affine** | **9** | **99.04%** | **99.09%** |

保持其他配置一致时，仿射增强让 MLP 测试准确率提高 0.61 个百分点，CNN 提高 0.38 个百分点。以上是同一划分、单种子结果。

- [PDF 实验报告](evidence/report.pdf) · [训练曲线](evidence/curves.png) · [权重图](evidence/parameters.png) · [混淆矩阵](evidence/confusion.png) · [预测错例](evidence/errors.png)
- [训练记录](evidence/train.json) · [测试记录](evidence/test.json) · [卷积梯度验算](evidence/verification.txt)
- [最佳 MLP 权重](evidence/mlp_affine.npz) · [最佳 CNN 权重](evidence/cnn_affine.npz)
