# 计算机视觉实验复现

各条目提供独立运行入口、实验设置、实测指标、可视化、报告和模型权重。原作业描述与原项目文件保留在原位置。

| 任务 | 复现入口 | 实测结果 |
|---|---|---|
| Fashion-MNIST 三层分类器 | [说明及产物](01-image-classification/fashion-mnist-3layer/reproduction/) | 7 组全数据集试验，官方测试准确率 88.60%（8,860 / 10,000） |
| MNIST MLP 与 CNN | [说明及产物](01-image-classification/nndl-mnist-mlp-cnn/reproduction/) | 6 组全数据集试验，最佳 MLP 98.57%，最佳 CNN 99.09% |
| CIFAR-10 VGG-A、BatchNorm 与损失景观 | [说明及产物](01-image-classification/nndl-cifar10-vgg/reproduction/) | 全宽 VGG-A 83.79%，VGG-A + BN 88.79%，VGG-A + Dropout 82.75% |
