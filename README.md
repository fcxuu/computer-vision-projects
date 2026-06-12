# Computer Vision Projects

A curated collection of computer vision projects covering image classification, object detection, self-supervised learning, Transformer vs CNN, and 3D reconstruction (NeRF).

> 来源说明：本仓库整合了以下 5 个旧仓库的内容（均已归档删除）：
> - `cxk049/computer-vision`
> - `cxk049/NNDL`
> - `cxk049/NNDL_PJ2`
> - `cxk049/midterm_project`
> - `cxk049/final_project`

## 目录结构

```
computer-vision-projects/
├── 01-image-classification/        # 图像分类
│   ├── fashion-mnist-3layer/       # Fashion-MNIST 三层神经网络
│   ├── nndl-mnist-mlp-cnn/         # MNIST MLP+CNN（神经网络与深度学习课程 PJ1）
│   └── nndl-cifar10-vgg/           # CIFAR-10 VGG + BatchNorm（神经网络与深度学习课程 PJ2）
├── 02-object-detection/            # 目标检测
│   └── voc-faster-rcnn-yolov3/     # VOC 数据集 Faster R-CNN + YOLO V3（含期中 Task 1 鸟类分类）
└── 03-self-supervised-and-transformer/  # 自监督学习 / 框架对比 / NeRF
    ├── cifar100-self-vs-supervised/  # CIFAR-100 自监督 vs 监督
    ├── cifar100-cnn-vs-transformer/  # CIFAR-100 CNN vs Transformer
    └── nerf-object-reconstruction/   # NeRF 物体重建（占位，原作业 Task 3 未完成）
```

每个子项目目录下都有：
- `HOMEWORK_ORIGINAL.txt` / `HOMEWORK_ORIGINAL.md` —— **原封不动** 的任务描述
- `README_SOURCE.md` —— 原仓库的 README 备份（如果有）
- 代码文件（.py / .ipynb）

---

## 01. 图像分类（Image Classification）

### [Fashion-MNIST 三层神经网络](01-image-classification/fashion-mnist-3layer/)
从零开始构建三层神经网络分类器，在 Fashion-MNIST 数据集上训练。
- 不使用 PyTorch/TensorFlow 等自动微分框架
- 纯 NumPy 手写反向传播
- 支持 SGD、学习率下降、交叉熵损失、L2 正则化、超参数搜索
- 包含训练好的模型权重 `trained_model.pkl`

### [MNIST MLP+CNN（NNDL PJ1）](01-image-classification/nndl-mnist-mlp-cnn/)
神经网络与深度学习课程 PJ1 —— MNIST 手写数字分类，**从零搭建神经网络框架**（仅用 NumPy）。
- MLP（784 → 600 → 10）
- CNN（Conv2D + ReLU + 全连接）
- SGD / MomentGD 优化器、StepLR / MultiStepLR / ExponentialLR
- 数据增强（旋转/平移/缩放）、权重可视化

### [CIFAR-10 VGG + BatchNorm（NNDL PJ2）](01-image-classification/nndl-cifar10-vgg/)
神经网络与深度学习课程 PJ2 —— CIFAR-10 分类，研究 VGG + BatchNorm + Loss Landscape。
- VGG-A / VGG-A-BN / VGG-A-Dropout 三种架构对比
- 损失景观可视化、梯度稳定性分析
- 详细训练策略与超参数调优

---

## 02. 目标检测（Object Detection）

### [VOC 数据集 Faster R-CNN + YOLO V3（期中作业）](02-object-detection/voc-faster-rcnn-yolov3/)
使用 mmdetection 框架在 PASCAL VOC 数据集上训练目标检测模型。
- **Task 1**：基于 ImageNet 预训练的 CNN（AlexNet/ResNet-18）在 CUB-200-2011 上微调做鸟类识别
  - ⚠️ 注：Task 1 实际上是**图像分类**任务（鸟类识别），与 Task 2（检测）同属一份期中作业，故放在本目录下
- **Task 2**：Faster R-CNN 和 YOLO V3 在 VOC 上训练与对比
  - 可视化 RPN 生成的 proposal box
  - 三张 VOC 类别外的图片做泛化测试

数据集和模型权重见原 Drive 链接：见子目录内 `HOMEWORK_ORIGINAL.txt`

---

## 03. 自监督学习 / 框架对比 / NeRF（Self-Supervised / Transformer / NeRF）

### [CIFAR-100 自监督 vs 监督（期末 Task 1）](03-self-supervised-and-transformer/cifar100-self-vs-supervised/)
对比监督学习和自监督学习在图像分类（CIFAR-100）上的性能。
- 任选一种自监督算法预训练 ResNet-18
- Linear Classification Protocol 在 CIFAR-100 上评测
- 与 ImageNet 监督预训练、从零训练三组对比

### [CIFAR-100 CNN vs Transformer（期末 Task 2）](03-self-supervised-and-transformer/cifar100-cnn-vs-transformer/)
基于 CNN 和 Transformer 架构的图像分类对比实验。
- 参数量相近的两种架构
- 数据增强包含 CutMix
- 详细的训练策略对齐

### [NeRF 物体重建（期末 Task 3 — 未完成）](03-self-supervised-and-transformer/nerf-object-reconstruction/)
> ⚠️ 原作业 Task 3 未完成（作者备注："由于时间问题，真的没时间写了，抱歉，过几天补上"）

按原任务描述：基于 NeRF 的物体重建和新视图合成 —— 多角度图片 + COLMAP 估计相机参数 + 训练 NeRF + 渲染视频。

此目录留作占位，待补完。

---

## 致谢 / 课程信息

本仓库的多数项目来自：
- **神经网络与深度学习**（NNDL）课程作业（PJ1 / PJ2）
- **计算机视觉**（Computer Vision）课程作业（期中 / 期末）

各项目的具体任务描述、数据集说明、提交要求均以各子目录下的 `HOMEWORK_ORIGINAL.*` 文件为准。
