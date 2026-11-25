# Faster R-CNN 的 Kaggle 双卡 T4 训练

Faster R-CNN 在 Kaggle 的两张 Tesla T4 上完成 8 轮训练。第 3 轮凭验证 mAP50 入选，官方测试为 **69.91% mAP50**，高于第一轮本机 Faster R-CNN 的 **63.05%**。训练代码、云端和本机复核指标分别保存在本目录。

## 对照设置

| 设置 | 第一轮本机 | Kaggle 正式训练 |
|---|---|---|
| 模型与初始化 | Faster R-CNN MobileNetV3-FPN，COCO | Faster R-CNN MobileNetV3-FPN，COCO |
| 输入与设备 | 256×256，Apple M1 Pro CPU | 320×320，两张 Tesla T4，DDP |
| 训练轮数与批量 | 5 轮，batch 4 | 8 轮，每卡 batch 8，全局名义 batch 16 |
| 训练翻转 | 随机水平翻转 | 无 |
| 验证 mAP50 | 61.16% | **68.82%** |
| 官方测试 mAP50 | 63.05% | **69.91%** |

两版共用 VOC2007 的 2,501 张训练图、2,510 张验证图和 4,952 张测试图。VOC07 AP 采用 IoU 0.50、11 点插值及 difficult 忽略规则。云端按验证 mAP50 选择，分数相同保留较早轮。正式测试的 20 类 AP 均高于第一轮。观察到的提升对应分辨率、批量、轮数与增强共同变化的整套配置。

控制台记录到官方测试结果的相对耗时为 653.04 秒。两张卡的 PyTorch 峰值分配分别为 463,808,512 和 462,759,936 字节，这个计数统计进程中的张量分配。云作业实际克隆的项目提交为 `87df4cbe8de15106c96caa2fba097012b6153138`，来自该作业输出中的 Git 克隆日志。

## 复现

本目录的 `train.py` 是从已完成的 Kaggle 作业拉取并校对过的原始提交代码。运行环境需要两张 CUDA GPU、PyTorch、TorchVision 和联网权限。要与此结果使用同一版项目代码，在 Kaggle 会话中先准备该提交。原脚本会直接使用现有的工作目录。

```bash
git clone --branch portfolio/reproductions-2026-11-23 https://github.com/fcxuu/computer-vision-projects.git /kaggle/working/computer-vision-projects
git -C /kaggle/working/computer-vision-projects checkout --detach 87df4cbe8de15106c96caa2fba097012b6153138
python /kaggle/working/train.py
```

提交脚本后，下载对应作业的结果时可以仅匹配目标文件。将变量设为你自己的任务标识。

```bash
kaggle kernels output "$KAGGLE_KERNEL" -p runs/voc-cloud --file-pattern '^faster-results/(best\.pt|best-val\.json|history\.json|test\.json)$'
```

在本机复核时，从仓库根目录运行如下命令。权重元数据包含 320 像素输入设置，由测试入口读取。

```bash
data/voc2007/.venv/bin/python 02-object-detection/voc-faster-rcnn-yolov3/reproduction/voc/experiment.py --device cpu test --batch 8 --checkpoint /path/to/faster-rcnn-320-t4.pt --metrics runs/voc-cloud-local-test.json --predictions runs/voc-cloud-local-predictions.npz
```

[云端指标与来源记录](evidence/) · [第一轮 VOC2007](../../evidence/report.pdf) · [短测对照](../README.md)。

## 本机复核与下载

下载的官方权重在本机 CPU 上回载，对 4,952 张图逐图推理得到 **69.91% mAP50**。独立从保存的预测框、类别和分数复算 VOC07 AP，结果一致。云端和本机的未舍入 mAP50 相差约 0.0001 个百分点。

[PDF 报告](evidence/report.pdf) · [学习曲线](evidence/learning-curves.png) · [逐类 AP](evidence/per-class-ap.png) · [本机复算指标](evidence/local-test.json) · [逐图预测](evidence/local-test-predictions.npz) · [模型权重](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-voc2007-cloud-2026-11-25) · [SHA-256](evidence/SHA256SUMS)。
