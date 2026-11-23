# 计算机视觉实验复现

各条目提供独立运行入口、实验设置、实测指标、可视化、报告和模型权重。原作业描述与原项目文件保留在原位置。

| 任务 | 复现入口 | 实测结果 |
|---|---|---|
| Fashion-MNIST 三层分类器 | [说明及产物](01-image-classification/fashion-mnist-3layer/reproduction/) | 7 组全数据集试验，官方测试准确率 88.60%（8,860 / 10,000） |
| MNIST MLP 与 CNN | [说明及产物](01-image-classification/nndl-mnist-mlp-cnn/reproduction/) | 6 组全数据集试验，最佳 MLP 98.57%，最佳 CNN 99.09% |
| CIFAR-10 VGG-A、BatchNorm 与损失景观 | [说明及产物](01-image-classification/nndl-cifar10-vgg/reproduction/) | 全宽 VGG-A 83.79%，VGG-A + BN 88.79%，VGG-A + Dropout 82.75% |
| CUB-200-2011 鸟类识别（期中任务 1） | [说明及产物](02-object-detection/voc-faster-rcnn-yolov3/reproduction/cub/) | ImageNet 微调 60.77% 对随机初始化 24.77%（官方测试 Top-1） |
| VOC2007 Faster R-CNN 与 YOLOv3（期中任务 2） | [说明及产物](02-object-detection/voc-faster-rcnn-yolov3/reproduction/voc/) | 官方测试 mAP50 分别为 63.05% 与 81.57%，含 4 张 RPN 和 3 张外部图对照 |
