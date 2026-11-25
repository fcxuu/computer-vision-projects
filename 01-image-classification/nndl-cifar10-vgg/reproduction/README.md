# CIFAR-10 · VGG-A、BatchNorm 与损失景观

比较 VGG-A、VGG-A + BN 和 VGG-A + Dropout 的真实训练结果，并用半宽模型对照参数量、学习率、激活、损失和优化器。三个主模型使用原 VGG-A 的卷积通道和五个池化阶段，只调整分类头以适应 CIFAR-10 的 32×32 输入。

## 本地复现

在本目录执行。默认优先使用可用的加速设备，这次实测使用 Apple M1 Pro 的 MPS。也可传 `--device cpu`。Torchvision 从官方来源下载并校验 CIFAR-10，默认使用仓库根目录的 `data/cifar10/`。已有官方数据可以用 `--data` 指定目录。

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate

python suite.py --device mps --out runs/full
python analyze.py --run runs/full --out runs/full/analysis --device mps
python report.py --run runs/full --evidence runs/full/analysis
tensorboard --logdir runs/full
```

逐个训练或从发布权重回载：

```bash
python experiment.py --device mps train --variant vgg_a_bn --width 1 --out runs/vgg-a-bn
mkdir -p runs
curl -fL --retry 2 -o runs/vgg-a-bn.pt https://github.com/fcxuu/computer-vision-projects/releases/download/portfolio-cifar10-vgg-2026-11-23/full-vgg_a_bn.pt
python experiment.py --device mps test --checkpoint runs/vgg-a-bn.pt
```

## 实验协议

- 从官方训练集按类划出 5,000 张验证图，余下 45,000 张训练。种子 2020，batch 128。只在训练时做反射填充随机裁剪和水平翻转。测试使用官方 10,000 张图片。
- 三个全宽模型和半宽 VGG-A 对照均训练 20 轮，使用 ReLU、交叉熵、AdamW、初始学习率 0.001、权重衰减 0.0001、余弦学习率调度。按最高验证准确率选权重，准确率并列时按验证交叉熵选。
- 在半宽 VGG-A 和半宽 VGG-A + BN 上各跑四组固定学习率 `0.0001`、`0.0005`、`0.001`、`0.002`。每组在全训练子集上跑三轮，记录每步的交叉熵和首层、输出层梯度范数，按同一训练步形成 min/max 损失包络。
- 半宽 VGG-A + BN 另用同预算比较 GELU、focal loss（γ=2）、更强权重衰减（0.001）和 SGD。SGD 使用学习率 0.03，其结果对应整个优化配置。不同训练损失的验证与评估统一使用交叉熵。
- 用两个全宽冠军权重在相同的固定验证子集上做二维滤波器归一化扰动，保留偏置、BN affine 参数和运行统计量。另沿中心梯度的下降方向测实际损失、一阶预测、梯度余弦和最大梯度差。它们反映所选方向和样本上的局部性质。

来源：[CIFAR-10 原始数据](https://www.cs.toronto.edu/~kriz/cifar.html) · [滤波器归一化损失景观论文](https://arxiv.org/abs/1712.09913) · [BN 优化分析论文](https://arxiv.org/abs/1805.11604)。

## 实测结果

| 模型 | 参数量 | 最佳验证轮 | 验证准确率 | 官方测试准确率 |
|---|---:|---:|---:|---:|
| VGG-A（半宽） | 2.44M | 20 | 82.94% | 82.59% |
| VGG-A | 9.75M | 20 | 84.12% | 83.79% |
| **VGG-A + BatchNorm** | **9.76M** | **18** | **89.44%** | **88.79%** |
| VGG-A + Dropout | 9.75M | 20 | 82.14% | 82.75% |

在本次全宽配对实验中，BN 组比普通 VGG-A 的官方测试准确率高 5.00 个百分点。Dropout 组比普通 VGG-A 低 1.04 个百分点。半宽 VGG-A 以约四分之一参数量得到 82.59% 测试准确率。所有结果使用一个固定种子。

- [PDF 实验报告](evidence/report.pdf) · [主训练曲线](evidence/curves.png) · [首层滤波器](evidence/filters.png)
- [学习率与梯度范数](evidence/lr-sweep.png) · [损失曲面](evidence/surface.png) · [梯度可预测性](evidence/gradient-probe.png) · [分析数值及样本索引](evidence/analysis.json)
- [逐轮与逐步日志，含主实验 TensorBoard 事件](evidence/) · [模型权重下载](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-cifar10-vgg-2026-11-23) · [SHA-256](evidence/SHA256SUMS)
