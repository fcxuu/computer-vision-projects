# SimCLR 长程预训练对照

第一轮在固定 1,200 步预训练预算下比较数据规模。最佳自监督冻结线性评估的官方测试 Top-1 为 30.43%。第二轮把同一套 SimCLR 训练延长到 4,800 步，冻结线性评估沿用第一轮方案。作品集目标为 40% 测试 Top-1。

## 运行

在本目录上一层运行。使用第一轮相同的依赖及 CIFAR-100 数据，设备为 Apple M1 Pro 的 MPS。

```bash
python experiment.py --device mps pretrain --variant 45k --steps 4800 --out runs/extended/pretrain-45k
python experiment.py --device mps train --source simclr_45k --pretrained runs/extended/pretrain-45k/pretrained.pt --out runs/extended/simclr_45k
```

配对组把命令中的 45k 改为 5k。每个编码器先按验证集选择最佳线性分类检查点，随后再运行测试。

```bash
python experiment.py --device mps test --checkpoint runs/extended/simclr_45k/best.pt --metrics runs/extended/simclr_45k/test.json
python optimization/report.py --baseline evidence --run runs/extended --out runs/extended/report
```

## 线性探测训练轮数

45,000 张组另比较 18 与 60 轮线性探测。两个预训练预算都使用相同的 60 轮探测设置。编码器始终冻结。

```bash
python experiment.py --device mps train --source simclr_45k --pretrained /path/to/first-round/pretrained-45k.pt --epochs 60 --out runs/extended/probe-60/1200
python experiment.py --device mps train --source simclr_45k --pretrained runs/extended/pretrain-45k/pretrained.pt --epochs 60 --out runs/extended/probe-60/4800
```

两组仍用验证 Top-1 选权重，并独立运行测试命令，将指标保存到各自的 test.json。

用同样两条命令加 `--standardize-probe`，并把输出目录改为 `standardized-60/1200` 与 `standardized-60/4800`，即可复现训练集统计标准化对照。均值、方差来自训练特征，导出时折叠成原坐标中的线性分类器。

## 45,000 张组继续预训练

从 4,800 步的编码器与投影头权重初始化，再跑 3,600 步。该阶段使用初始学习率 0.0004 的新余弦调度、新 AdamW 状态及独立增强随机序列。

```bash
python experiment.py --device mps pretrain --variant 45k --steps 3600 --lr 0.0004 --initialize-from runs/extended/pretrain-45k/pretrained.pt --out runs/extended/continuation-45k
python experiment.py --device mps train --source simclr_45k --pretrained runs/extended/continuation-45k/pretrained.pt --epochs 60 --out runs/extended/continuation-probe
```

比较固定 60 轮原始特征探测下的 4,800 和累计 8,400 步权重。

## 对照设置

同图像池的两轮共用原始训练、验证索引和相同种子的预训练取样。每步 batch 128，两视图做 NT-Xent，温度 0.2，输入 64×64。AdamW 初始学习率 0.001，沿用余弦衰减形式并按总步数重新铺开调度。4800 步组从初始化开始训练。线性评估固定训练 18 轮、同一划分、同一归一化和优化设置，并校验编码器参数与 BN 缓冲区在探测前后的指纹。报告分开记录预训练配对检索率和官方分类测试准确率。

[第一轮报告](../evidence/report.pdf) · [SimCLR 论文](https://proceedings.mlr.press/v119/chen20j.html)。

## 实测结果

| 图像池 | 预训练步数 | 线性探测 | 验证 Top-1 | 官方测试 Top-1 | 官方测试 Top-5 |
|---|---:|---|---:|---:|---:|
| 45,000 | 1,200 | 原始特征，18 轮 | 30.76% | 29.79% | 57.63% |
| 45,000 | 4,800 | 原始特征，18 轮 | 37.54% | 36.44% | 65.79% |
| 5,000 | 1,200 | 原始特征，18 轮 | 30.88% | 30.43% | 57.83% |
| 5,000 | 4,800 | 原始特征，18 轮 | 38.28% | 37.51% | 66.19% |
| 45,000 | 1,200 | 原始特征，60 轮 | 32.72% | 31.34% | 58.66% |
| 45,000 | 4,800 | 原始特征，60 轮 | 38.84% | 38.07% | 65.99% |
| 45,000 | 1,200 | 标准化特征，60 轮 | 32.48% | 31.39% | 58.07% |
| 45,000 | 4,800 | 标准化特征，60 轮 | 38.00% | 37.36% | 64.81% |
| 45,000 | 8,400 | 原始特征，60 轮 | **39.64%** | **39.47%** | **67.60%** |

验证集选出的配置在第 58 轮线性探测答对官方测试集的 3,947 / 10,000 张图，较第一轮最佳 SimCLR 提高 9.04 个百分点，距 40% 的目标还差 0.53 个百分点。固定原始特征的 60 轮探测时，4,800 至累计 8,400 步的续训方案提高 1.40 个百分点。标准化在两个编码器上都没有提高验证 Top-1。

- [第二轮报告](evidence/report.pdf) · [预训练预算曲线](evidence/budget-curves.png) · [线性探测曲线](evidence/probe-curves.png) · [续训曲线](evidence/continuation-curves.png)
- [逐组训练记录、指标和 TensorBoard](evidence/) · [模型权重](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-simclr-extended-2026-11-24) · [SHA-256](evidence/SHA256SUMS)
