# CNN 与 ViT 的 40 轮配对实验

第一轮的匹配设置在 CIFAR-100 官方测试集获得 CNN 59.09%、ViT 46.20% Top-1。第二轮把两种网络的训练预算从 20 轮延长为 40 轮，从随机初始化开始。ViT 的目标为官方测试 Top-1 52%。

## 复现

在本目录上一层执行，使用第一轮的 CIFAR-100 数据和 Python 环境。

```bash
python experiment.py --device mps train --variant cnn --setting adamw_1e-3 --epochs 40 --out runs/extended/cnn-adamw_1e-3
python experiment.py --device mps train --variant vit --setting adamw_1e-3 --epochs 40 --out runs/extended/vit-adamw_1e-3
python experiment.py --device mps test --checkpoint runs/extended/cnn-adamw_1e-3/best.pt --metrics runs/extended/cnn-adamw_1e-3/test.json
python experiment.py --device mps test --checkpoint runs/extended/vit-adamw_1e-3/best.pt --metrics runs/extended/vit-adamw_1e-3/test.json
python optimization/report.py --baseline evidence --run runs/extended --out runs/extended/report
```

## 对照设置

- CNN 有 871,732 个可训练参数，ViT 有 861,060 个，以较小模型为分母相差 1.24%。共用 45,000 张训练图、5,000 张验证图及 10,000 张官方测试图。
- 两模型使用 batch 256、AdamW、初始学习率 0.001、权重衰减 0.03、两轮预热、余弦学习率衰减及梯度裁剪 5。余弦时间轴随训练总轮数延长。
- 沿用随机裁剪、翻转和 CutMix。相同种子使配对组逐轮经历相同的 CutMix 批次数与替换面积。验证及测试使用未增强图像。
- 每个架构用验证 Top-1 选最佳权重，同分时用验证交叉熵。两组训练完成后回载权重，统一测官方测试集。

[第一轮实验报告](../evidence/report.pdf) · [CIFAR-100 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)。

## 实测结果

| 模型 | 20 轮测试 Top-1 | 40 轮最佳验证 Top-1 | 40 轮测试 Top-1 | 40 轮测试 Top-5 |
|---|---:|---:|---:|---:|
| CNN | 59.09% | 63.86% | **63.27%** | 87.82% |
| ViT | 46.20% | 51.26% | **51.26%** | 79.44% |

两架构均选中第 37 轮，实测测试 Top-1 分别提高 4.18 和 5.06 个百分点。ViT 距 52% 目标 0.74 个百分点。每组 40 轮共执行 7,040 次更新，3,466 个批次启用了 CutMix，占 49.23%。这些结果基于一个固定随机种子与一套匹配优化配置。

[第二轮 PDF](evidence/report.pdf) · [学习曲线](evidence/budget-curves.png) · [原始日志与 TensorBoard](evidence/) · [权重发布](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-cifar100-arch-extended-2026-11-24) · [SHA-256](evidence/SHA256SUMS)。
