# Faster R-CNN 混合设备短测

2026 年 11 月 24 日，在 Apple M1 Pro 上比较纯 CPU 与混合 MPS／CPU。混合路径将外层模型和梯度更新放在 MPS，只把 ROIAlign 放在 CPU。两边都严格加载同一份第一轮 VOC2007 训练权重，再在计划用于优化的 320 像素输入上测量。

## 方法

两个回合交换设备顺序。每个回合使用固定且逐图核对过的真实 VOC 批次，batch 4，每个设备预热 2 步、计时 6 步。预解码在计时外。计时包括翻转、设备传输、目标转换、计算训练损失、反向传播、梯度裁剪和 SGD 更新。MPS 在计时前后同步。另在相同的小批验证图上检查验证损失、预测框和 RPN 候选框。

## 结果

| 配对轮次 | 纯 CPU 步耗时中位数 | 混合 MPS／CPU 步耗时中位数 | 混合步耗时变化 |
|---|---:|---:|---:|
| 1 | 0.842 秒 | 1.057 秒 | +25.6% |
| 2 | 0.775 秒 | 1.156 秒 | +49.2% |

两种方式的训练损失、梯度和验证输出通过有限值检查，均发生了参数更新。混合方式两轮都较慢，本轮到短测结束，正式纯 CPU 全量训练留到晚上。上表反映本机、这套模型与批量设置下的短测性能。

## 复现

在仓库根目录执行。使用 VOC 第一轮的虚拟环境、数据和 Faster R-CNN 验证最佳检查点。

```bash
data/voc2007/.venv/bin/python 02-object-detection/voc-faster-rcnn-yolov3/reproduction/voc/optimization/benchmark_hybrid.py --image-size 320 --batch 4 --warm-up 2 --steps 6 --out runs/voc2007-hybrid-check
```

[原始逐步计时、训练与验证检查数据](evidence/gate.json) · [第一轮 VOC2007 结果](../evidence/report.pdf)。
