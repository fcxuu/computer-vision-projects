# Fashion-MNIST · NumPy 三层分类器

本次复现使用 784 → H → 10 的网络。两层仿射变换、激活函数、逐样本稳定 softmax、交叉熵和反向传播全部用 NumPy 实现。隐藏层支持 ReLU、sigmoid、tanh。小批量 SGD 每轮衰减学习率，反传时仅对权重加 L2 梯度。

## 运行

在本目录执行。脚本默认从 Fashion-MNIST 官方仓库下载 IDX 压缩包到仓库根的 `data/fashion-mnist/`，并校验发布方提供的 MD5。已有数据可用 `--data /path/to/fashion-mnist` 指定，放在子命令前。输出目录须是新路径，以便保留各次运行的结果。

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
export VECLIB_MAXIMUM_THREADS=4 OPENBLAS_NUM_THREADS=4

python verify.py
python experiment.py search --out runs/full
python experiment.py test --checkpoint runs/full/best.npz --metrics runs/full/test.json
python report.py --run runs/full --out runs/full/report
```

如果只想试一个配置：

```bash
python experiment.py train --out runs/custom --hidden 128 --activation tanh \
  --lr 0.08 --l2 0.0001 --epochs 36 --seed 23
python experiment.py test --checkpoint evidence/best.npz
```

## 实验协议

- 官方训练集按类别固定划分为 54,000 张训练图与 6,000 张验证图。所有试验使用同一划分。每类 600 张验证图。像素除以 255。
- 七组试验统一训练 36 轮，batch 256，每轮学习率乘 0.97。相对基准分别改变初始学习率、隐藏层宽度、L2 强度或激活函数。
- 每组试验按验证准确率保存最佳轮，准确率并列时比较验证交叉熵，再并列时保留较早轮。组间沿用同一组指标，完全并列时保留先运行的配置。训练命令仅读取官方训练文件。
- 独立的 `test` 命令从冻结的冠军 NPZ 恢复参数，在官方 10,000 张测试图上计算交叉熵、准确率和混淆矩阵。训练和验证曲线使用无 L2 的交叉熵以便比较，CSV 另记录训练目标值。
- 梯度验算覆盖三种激活函数及两档 L2。回载检查比较权重保存前后的 logits，训练脚本还会重算冠军权重的验证指标。

数据来源：[Zalando Research Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)。

## 实测结果

| 配置 | 最佳验证准确率 |
|---|---:|
| ReLU 基准，H=256，学习率 0.12，L2=0.0001 | 88.65% |
| 学习率 0.06 | 87.35% |
| **学习率 0.24** | **89.58%** |
| H=128 | 88.45% |
| H=512 | 88.95% |
| L2=0.001 | 88.45% |
| sigmoid 隐层 | 84.93% |

验证集选中的 **第 34 轮** 在官方测试集上答对 **8,860 / 10,000**，测试准确率 **88.60%**，测试交叉熵 **0.32343**。这些比较来自固定划分和单个随机种子。

- [PDF 实验报告](evidence/report.pdf)
- [训练与验证曲线](evidence/learning-curves.png) · [参数可视化](evidence/parameters.png) · [测试混淆矩阵](evidence/confusion.png)
- [搜索记录](evidence/search.json) · [测试结果](evidence/test.json) · [验算输出](evidence/verification.txt)
- [最佳模型权重](evidence/best.npz)，SHA-256 `dd0d17f8622e3c11d3ad621c7694304ffe2f913f2c4f397fb655be09230a9b41`
