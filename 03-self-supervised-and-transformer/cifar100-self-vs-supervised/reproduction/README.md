# CIFAR-100 · 自监督 SimCLR 与监督学习

复现计算机视觉期末任务 1。使用同一 ResNet-18 架构，对比 SimCLR 表征、ImageNet 监督预训练表征与 CIFAR-100 随机初始化监督训练。所有指标以本机实际训练与官方测试为准。

## 运行

在本目录执行。默认使用仓库根的 `data/cifar100/`，首次训练由 Torchvision 下载并校验官方归档。已有 `cifar-100-python.tar.gz` 可放入该目录，官方 MD5 为 `eb9058c3a382ffc7106e4002c42a8d85`。

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate

python suite.py --device mps --out runs/full
python report.py --run runs/full --out runs/full/report
tensorboard --logdir runs/full
```

也可逐步执行一组实验：

```bash
python experiment.py --device mps pretrain --variant 20k --out runs/pretrain-20k
python experiment.py --device mps train --source simclr_20k --pretrained runs/pretrain-20k/pretrained.pt --out runs/probe-20k
python experiment.py --device mps test --checkpoint runs/probe-20k/best.pt --metrics runs/probe-20k/test.json
```

## 实验设计

- 官方训练集按类留出 5,000 张验证图，每类 50 张，余下 45,000 张作训练。官方 10,000 张测试图用于最终评估。使用种子 2020，三组数据规模按同一随机排列取前缀，互相包含。
- SimCLR 只读取训练图像，每个样本经独立随机裁剪、翻转、颜色扰动和灰度化生成两视图。原标签不参与预训练损失。ResNet-18 接 512 → 512 → 128 投影头。使用 NT-Xent，正样本为同图像的两视图，其他图像的视图为负样本。
- 固定 batch 128、1,200 次优化更新与 64×64 输入，比较 5,000、20,000、45,000 张源图。数据规模对照使用温度 0.2，20,000 张另测温度 0.5。相同步数固定计算预算，较小子集的样本被重复采样更多次。
- ImageNet 监督权重与 SimCLR 都冻结整个 ResNet-18，保持评估模式并固定 BN 统计量。用同样的干净图像和同一配置训练 512 → 100 线性层 18 轮。训练前后校验编码器参数及缓冲区的 SHA-256。
- 随机初始化组在同样的 CIFAR-100 训练划分上更新全部网络参数。监督训练与线性协议均使用交叉熵，按验证 Top-1 保存，准确率相同时按验证损失取较低者。所有测试用各自冻结的最佳权重。

来源：[CIFAR-100 官方数据](https://www.cs.toronto.edu/~kriz/cifar.html) · [SimCLR 原论文](https://arxiv.org/abs/2002.05709)。

## 实测结果

| 训练方式 | 最佳轮 | 验证 Top-1 | 官方测试 Top-1 | 官方测试 Top-5 |
|---|---:|---:|---:|---:|
| 随机初始化端到端 | 18 | 57.98% | 57.49% | 83.23% |
| ImageNet 监督预训练 | 17 | 51.42% | 51.40% | 81.62% |
| SimCLR 5,000 | 18 | 30.88% | 30.43% | 57.83% |
| SimCLR 20,000 | 18 | 29.66% | 29.73% | 57.27% |
| SimCLR 45,000 | 18 | 30.76% | 29.79% | 57.63% |
| SimCLR 20,000（温度 0.5） | 18 | 25.72% | 25.48% | 52.32% |

验证集选中的自监督配置是 **SimCLR 5,000**，官方测试 Top-1 为 **30.43%**。在本次固定 1,200 步的预算下，扩大预训练图像池没有带来单调的 Top-1 提升。随机初始化组用类别标签训练全网，ImageNet 和 SimCLR 两组属于冻结线性评估。

- [PDF 实验报告](evidence/report.pdf) · [预训练曲线](evidence/pretrain-curves.png) · [分类学习曲线](evidence/classification-curves.png)
- [日志和 TensorBoard](evidence/) · [数据范围核对](evidence/dataset.json) · [代码验算](evidence/verification.txt)
- [预训练和分类器权重](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-cifar100-simclr-2026-11-23) · [SHA-256](evidence/SHA256SUMS)
