# 计算机视觉实验复现

各条目提供独立运行入口、实验设置、实测指标、可视化、报告和模型权重。原作业描述与原项目文件保留在原位置。

[中文阶段报告（已完成七项）](PORTFOLIO_REPORT_2026-11-23.pdf)

| 任务 | 复现入口 | 实测结果 |
|---|---|---|
| Fashion-MNIST 三层分类器 | [说明及产物](01-image-classification/fashion-mnist-3layer/reproduction/) | 7 组全数据集试验，官方测试准确率 88.60%（8,860 / 10,000） |
| MNIST MLP 与 CNN | [说明及产物](01-image-classification/nndl-mnist-mlp-cnn/reproduction/) | 6 组全数据集试验，最佳 MLP 98.57%，最佳 CNN 99.09% |
| CIFAR-10 VGG-A、BatchNorm 与损失景观 | [说明及产物](01-image-classification/nndl-cifar10-vgg/reproduction/) | 全宽 VGG-A 83.79%，VGG-A + BN 88.79%，VGG-A + Dropout 82.75% |
| CUB-200-2011 鸟类识别（期中任务 1） | [说明及产物](02-object-detection/voc-faster-rcnn-yolov3/reproduction/cub/) | ImageNet 微调 60.77% 对随机初始化 24.77%（官方测试 Top-1） |
| VOC2007 Faster R-CNN 与 YOLOv3（期中任务 2） | [说明及产物](02-object-detection/voc-faster-rcnn-yolov3/reproduction/voc/) | 官方测试 mAP50 分别为 63.05% 与 81.57%，含 4 张 RPN 和 3 张外部图对照 |
| CIFAR-100 自监督与监督学习（期末任务 1） | [说明及产物](03-self-supervised-and-transformer/cifar100-self-vs-supervised/reproduction/) | 官方测试 Top-1 为随机初始化 57.49%、ImageNet 冻结线性 51.40%、最佳 SimCLR 冻结线性 30.43% |
| CIFAR-100 CNN 与 Transformer（期末任务 2） | [说明及产物](03-self-supervised-and-transformer/cifar100-cnn-vs-transformer/reproduction/) | 参数量差 1.24%，验证选型后的官方测试 Top-1 为 CNN 59.09%、ViT 46.20% |
| NeRF 物体重建与新视角合成（期末任务 3） | [说明及产物](03-self-supervised-and-transformer/nerf-object-reconstruction/reproduction/) | COLMAP 注册 67 / 67 张。留出视角 17.57 dB PSNR、0.4819 SSIM，含 90 帧环绕视频 |
