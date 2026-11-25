# CUB-200-2011 · ResNet-18 鸟类识别

复现计算机视觉期中任务 1。ResNet-18 的输出层改为 200 类，比较 ImageNet 权重微调与随机初始化。该任务的数据、权重和指标独立于同一作业的 VOC 检测任务。

## 运行

在本目录执行。先从 [CaltechDATA](https://data.caltech.edu/records/65de6-vp158) 下载 `CUB_200_2011.tgz`。默认数据目录是仓库根的 `data/cub200/`，也可显式指定。程序按官方 MD5 `97eceeb196236b17998738112f37df78` 校验归档，安全解包；可复用含完整官方元数据的已解包目录。训练和测试命令在本地按划分缓存各自的 RGB 张量。

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate

python suite.py --device mps --data /path/to/cub200 --out runs/full
python report.py --run runs/full --out runs/full/report
tensorboard --logdir runs/full
```

也可单独训练或加载发布权重跑官方测试：

```bash
python experiment.py --device mps train --trial transfer --out runs/transfer
mkdir -p runs
curl -fL --retry 2 -o runs/best-transfer.pt https://github.com/fcxuu/computer-vision-projects/releases/download/portfolio-cub200-2026-11-23/best-transfer.pt
python experiment.py --device mps test --checkpoint runs/best-transfer.pt
```

## 实验设置

- 保留官方 5,794 张测试图。验证图从官方 5,994 张训练图中按类随机留出约 15%，划分种子为 2020。所有试验共用划分及分类头的随机初始化。
- 图像统一缩放至 160×160，按 ImageNet 均值方差归一化。训练时做随机裁剪和平移及水平翻转，验证与测试使用干净的缩放图。
- 训练 16 轮，batch 48。所有组均使用交叉熵、AdamW、权重衰减 0.0001 和余弦学习率调度。微调组第一轮冻结骨干及其 BN 统计量，之后用更小的学习率训练骨干。随机初始化组从第一轮更新全网。
- 使用验证 Top-1 保存每组最优轮次，准确率相同时取较低验证交叉熵。从三个微调配置中按同一准则选中一个，随后在官方测试集比较 Top-1、Top-5 和按类平均召回率。

| 配置 | 初始化 | 骨干学习率 | 分类头学习率 |
|---|---|---:|---:|
| scratch | 随机 | 0.001 | 0.002 |
| transfer | ImageNet | 0.0002 | 0.002 |
| transfer_low | ImageNet | 0.00007 | 0.001 |
| transfer_high | ImageNet | 0.0005 | 0.003 |

来源：[CUB-200-2011 官方数据](https://doi.org/10.22002/D1.20098) · [Torchvision ResNet-18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。

## 实测结果

| 配置 | 最佳验证轮 | 验证 Top-1 | 测试 Top-1 | 测试 Top-5 | 宏平均召回率 |
|---|---:|---:|---:|---:|---:|
| scratch | 15 | 24.14% | 24.77% | 52.59% | 25.00% |
| transfer | 16 | 59.51% | 60.77% | 85.61% | 61.10% |
| transfer_low | 13 | 59.07% | 60.13% | 86.26% | 60.43% |
| transfer_high | 15 | 58.62% | 60.17% | 84.40% | 60.38% |

验证集选中 **transfer**。该配置在官方测试集上答对 **3,521 / 5,794**，Top-1 比随机初始化高 **36.00 个百分点**。

- [实验报告](evidence/report.pdf) · [训练曲线](evidence/curves.png) · [错分对比](evidence/test-errors.png)
- [训练、测试和 TensorBoard 原始指标](evidence/) · [数据划分验收记录](evidence/dataset.json)
- [模型权重下载](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-cub200-2026-11-23) · [SHA-256](evidence/SHA256SUMS)
