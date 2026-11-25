# CIFAR-100 · CNN 与视觉 Transformer

复现计算机视觉期末任务 2。分别实现残差 CNN 和 ViT，在统一数据、参数预算和训练策略下比较分类表现。CNN 有 871,732 个可训练参数，ViT 有 861,060 个，以较小模型为基数相差 1.24%。

- CNN 用 48 通道 stem、64/96/128/144 通道残差块和全局平均池化。每个块包含两层 3×3 卷积，有下采样或通道数变化时通过投影连接捷径。
- ViT 用 4×4 patch，把 32×32 输入转换成 64 个图像 token，加入 CLS 和可学习位置嵌入。4 个 pre-LayerNorm 块，嵌入宽 160，每层 4 个注意力头，MLP 隐层宽 320，激活为 GELU。

## 运行

从仓库完整检出本分支，在本目录运行。数据加载复用 [CIFAR-100 数据管线](../../cifar100-self-vs-supervised/reproduction/cifar100_data.py)。默认数据位于仓库根的 `data/cifar100/`。首次训练由 Torchvision 校验和解包。

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate

python verify.py
python suite.py --device mps --out runs/full
python report.py --run runs/full --out runs/full/report
tensorboard --logdir runs/full
```

单独训练和测试：

```bash
python experiment.py --device mps train --variant vit --setting adamw_1e-3 --out runs/vit
python experiment.py --device mps test --checkpoint runs/vit/best.pt --metrics runs/vit/test.json
```

## 实验协议

- 45,000 张训练图、5,000 张分层验证图和 10,000 张官方测试图。训练划分种子 2020，输入 32×32，训练时做随机裁剪、翻转和 CutMix，验证、测试使用干净图像。
- 训练 20 轮，每批 256 张，全部组使用 AdamW、两轮预热和余弦学习率衰减，梯度裁剪 5。两架构分别运行初始学习率 0.001 与权重衰减 0.03，以及初始学习率 0.0005 与权重衰减 0.01。
- CutMix 以 0.5 的概率在批次中拼接同一随机矩形，面积比例来自 Beta(1,1)，按裁剪后实际面积混合标签损失。训练准确率也按两组标签权重加权。验证和测试统一用普通交叉熵与 Top-1/Top-5。
- 先按最高验证准确率保存各次运行的权重，并列时比较验证交叉熵。每个架构也按同一规则选配置，最后比较对应官方测试结果。

来源：[CIFAR-100 官方数据](https://www.cs.toronto.edu/~kriz/cifar.html) · [CutMix 论文](https://arxiv.org/abs/1905.04899) · [Vision Transformer 论文](https://arxiv.org/abs/2010.11929)。

## 实测结果

| 模型 | 初始学习率 | 权重衰减 | 最佳验证轮 | 验证 Top-1 | 测试 Top-1 | 测试 Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.001 | 0.03 | 20 | 59.80% | 59.09% | 85.74% |
| ViT | 0.001 | 0.03 | 19 | 45.32% | 46.20% | 75.90% |
| CNN | 0.0005 | 0.01 | 20 | 55.68% | 55.37% | 83.38% |
| ViT | 0.0005 | 0.01 | 20 | 42.86% | 43.65% | 74.15% |

两架构均以验证集选中初始学习率 0.001、权重衰减 0.03 的配置。在本次相近参数量与统一训练预算下，CNN 的官方测试 Top-1 比 ViT 高 12.89 个百分点。两套配对实验的逐轮 CutMix 批次数和平均替换面积均一致。

- [PDF 实验报告](evidence/report.pdf) · [训练和验证曲线](evidence/curves.png) · [指标及 TensorBoard](evidence/)
- [模型权重](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-cifar100-architectures-2026-11-23) · [SHA-256](evidence/SHA256SUMS) · [功能验算](evidence/verification.txt)
