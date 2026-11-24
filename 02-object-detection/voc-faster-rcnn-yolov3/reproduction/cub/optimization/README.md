# CUB-200-2011 鸟类识别第二轮优化

第一轮 ResNet-18 微调在官方测试集取得 60.77% Top-1。第二轮沿用同一个模型、官方划分和随机种子，把输入提高到 224 像素，采用等比补边和标签平滑。测试目标为 65%，争取 68%。

## 复现

在本目录上一层执行，沿用第一轮依赖与 CUB 数据。

```bash
python experiment.py --device mps train --trial transfer --out runs/extended/transfer-224 --image-size 224 --resize-mode letterbox --label-smoothing 0.1 --epochs 24 --batch 48
python experiment.py --device mps validate --checkpoint runs/extended/transfer-224/best.pt --metrics runs/extended/transfer-224/validation-plain.json
python experiment.py --device mps validate --checkpoint runs/extended/transfer-224/best.pt --metrics runs/extended/transfer-224/validation-flip.json --flip-tta
```

用验证 Top-1 选择单视图或水平翻转平均，准确率相同时用验证交叉熵，两者完全相同则保留单视图。冻结后从对应检查点运行 `test`，仅在选中翻转平均时附加 `--flip-tta`，输出指标到 `runs/extended/transfer-224/test.json`。

```bash
python optimization/report.py --baseline evidence --run runs/extended --out runs/extended/report
```

## 对照设置

- 5,095 张训练、899 张验证及 5,794 张官方测试图。等比缩放的最长边为 224，居中使用近似 ImageNet 通道均值的 RGB 补边，以保留完整视野。
- 使用 ResNet-18 ImageNet 权重初始化。第一轮冻结骨干，之后使用 AdamW 分组学习率，骨干 0.0002、分类头 0.002，权重衰减 0.0001。余弦衰减覆盖 24 轮。
- 训练损失使用 0.1 标签平滑。验证和测试交叉熵使用硬标签。该实验联合改变输入、预处理、训练预算和训练损失。
- Caltech 提示 CUB 与 ImageNet 原始图像可能重叠，解释迁移学习结果时需保留这一条件。[CUB-200-2011 官方说明](https://www.vision.caltech.edu/datasets/cub_200_2011/)。

[第一轮实验报告](../evidence/report.pdf)。

## 实测结果

第 24 轮入选。该权重的单视图验证 Top-1 为 **66.18%**，翻转平均为 **66.85%**，故官方测试采用翻转平均。

| 检查点 | 官方测试 Top-1 | 官方测试 Top-5 | 宏平均召回率 |
|---|---:|---:|---:|
| 第一轮，160 像素 | 60.77%（3,521 / 5,794） | 85.61% | 61.10% |
| 第二轮，224 像素 | **68.26%（3,955 / 5,794）** | **87.68%** | **68.51%** |

官方测试 Top-1 提高 7.49 个百分点，达到 65% 的目标和 68% 的进阶目标。训练与验证的索引顺序逐项一致，测试类支持数逐项一致。

[第二轮 PDF](evidence/report.pdf) · [验证学习曲线](evidence/validation-curves.png) · [类别召回变化](evidence/class-differences.png) · [原始记录与 TensorBoard](evidence/) · [权重发布](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-cub224-2026-11-24) · [SHA-256](evidence/SHA256SUMS)。
