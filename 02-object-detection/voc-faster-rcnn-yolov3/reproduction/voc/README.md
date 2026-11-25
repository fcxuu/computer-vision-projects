# VOC2007 · Faster R-CNN 与 YOLOv3

复现计算机视觉期中任务 2。使用 Torchvision Faster R-CNN MobileNetV3-Large FPN 和 Ultralytics 原版三尺度有锚框 YOLOv3（Darknet53），统一在 PASCAL VOC2007 上训练和评估。两者从各自的 COCO 权重开始，将共享类别的检测头行映射到 VOC 20 类，随后在 VOC 图像上更新模型。Faster R-CNN 保留背景类。

## 数据与环境

把官方 [VOC2007 训练验证归档](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) 和 [带标注的测试归档](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) 放到仓库根目录的 `data/voc2007/`。加载器自动校验 MD5 并分开解包，训练入口只打开训练验证归档。实测使用 VOC2007 官方划分，train 2,501 张、val 2,510 张、test 4,952 张。

从仓库根目录取得固定的 YOLOv3 依赖和发布权重：

```bash
mkdir -p data/vendor data/voc2007/pretrained
git clone https://github.com/ultralytics/yolov3.git data/vendor/yolov3
git -C data/vendor/yolov3 checkout --detach 34c3f2108b73031647b293a7e73603d90e5a208b
curl -fL --retry 2 -o data/voc2007/pretrained/yolov3.pt https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt
```

接下来在本复现目录执行。采用独立的 Python 3.12 环境，以使用 YOLOv3 依赖所需的 NumPy 版本。Torchvision 会下载 Faster R-CNN 的 COCO 权重。YOLOv3 权重加载前校验本次使用的 SHA-256。

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate

python verify_voc.py
python experiment.py --device cpu train --variant faster_rcnn --size 256 --epochs 5 --batch 4 --val-batch 8 --out runs/full/faster_rcnn
python experiment.py --device mps train --variant yolov3 --size 256 --epochs 5 --batch 8 --val-batch 8 --out runs/full/yolov3
```

## 测试与可视化

```bash
python experiment.py --device cpu test --checkpoint runs/full/faster_rcnn/best.pt --metrics runs/full/faster_rcnn/test.json --predictions runs/full/faster_rcnn/test-predictions.npz
python experiment.py --device mps test --checkpoint runs/full/yolov3/best.pt --metrics runs/full/yolov3/test.json --predictions runs/full/yolov3/test-predictions.npz
python visualize.py --device cpu --faster runs/full/faster_rcnn/best.pt --yolo runs/full/yolov3/best.pt --out runs/full/figures
python report.py --run runs/full --out runs/full/figures
tensorboard --logdir runs/full
```

- 两模型共享 256×256 等比缩放加居中填充。训练时随机水平翻转，测试坐标反变换回原图。训练损失忽略标注为 `difficult` 的框。
- Faster R-CNN 的损失含 RPN 和 ROI 分类、回归项。YOLOv3 的损失含类别、目标置信度与 CIoU 框回归。两模型都冻结部分早层和 BN 统计量，使用 SGD、动量 0.9、权重衰减 0.0001、梯度裁剪 10 和余弦退火。初始学习率分别为 0.003 和 0.009。
- 每轮记录训练和验证损失及验证 mAP50。使用 VOC2007 的 11 点 AP，IoU ≥ 0.50，忽略与 `difficult` 真值匹配的检测，并且每个有效真值最多匹配一次。根据验证 mAP50 保存权重，并列时取更小的同模型验证损失。
- 四张测试图的 RPN 图使用真正的 RPN 前向输出，按序取前 24 个候选。外部对照图使用相同的 0.20 显示阈值。三张实拍图的来源、署名与许可见 [样图说明](examples/SOURCES.md)。

来源：[PASCAL VOC2007](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/) · [Torchvision Faster R-CNN](https://docs.pytorch.org/vision/stable/models/faster_rcnn.html) · [固定的原版 YOLOv3 配置](https://github.com/ultralytics/yolov3/blob/34c3f2108b73031647b293a7e73603d90e5a208b/models/yolov3.yaml)。

## 实测结果

| 模型 | 本机设备 | 最佳验证轮 | 验证 mAP50 | 官方测试 mAP50 |
|---|---|---:|---:|---:|
| Faster R-CNN MobileNetV3-FPN | CPU | 5 | 61.16% | 63.05% |
| YOLOv3 Darknet53 | MPS | 5 | 80.70% | 81.57% |

在这两套预训练、架构和训练配置下，YOLOv3 的测试 mAP50 高 18.52 个百分点。测试集共 4,952 张图。数据核对显示三个官方划分之间没有重复 ID 或完全相同的图像文件。可用已发布的逐图检测坐标重新计算 20 类 AP。

- [PDF 实验报告](evidence/report.pdf) · [训练曲线](evidence/curves.png) · [逐类 AP 对比](evidence/per-class.png)
- [四张测试图的 RPN 和最终预测](evidence/rpn.png) · [自行车](evidence/bicycle-annotated.jpg) · [猫](evidence/cat-annotated.jpg) · [牛](evidence/cow-annotated.jpg) · [框与置信度原始记录](evidence/visualizations.json)
- [逐轮/逐图日志和 TensorBoard](evidence/) · [划分核对](evidence/dataset.json) · [权重下载](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-voc2007-2026-11-23) · [SHA-256](evidence/SHA256SUMS)
