# VOC 数据集 Faster R-CNN + YOLO V3（期中作业）

**来源仓库**：`cxk049/midterm_project`（已归档）

## 项目简介

计算机视觉课程期中作业 —— 包含两个 Task：

### Task 1：基于 ImageNet 预训练的鸟类识别

微调在 ImageNet 上预训练的卷积神经网络（AlexNet / ResNet-18）实现 CUB-200-2011 数据集上的鸟类识别（200 类细粒度分类）。

### Task 2：VOC 数据集目标检测

使用 mmdetection 框架在 PASCAL VOC 数据集上训练并测试目标检测模型 Faster R-CNN 和 YOLO V3，并进行可视化对比。

> ⚠️ Task 1 实际上是**图像分类**任务（鸟类识别），与 Task 2（检测）同属一份期中作业，故放在本目录。

## 文件结构

```
voc-faster-rcnn-yolov3/
├── homework_task_1.ipynb       # Task 1：鸟类识别
├── homework_task_2.ipynb       # Task 2：Faster R-CNN + YOLO V3 检测
├── HOMEWORK_ORIGINAL.txt       # 原任务描述（保留原文）
├── README_SOURCE.md            # 原仓库 README 备份
└── README.md                   # 本文件
```

## 使用说明

### Task 1（鸟类识别）

打开 `homework_task_1.ipynb`，按顺序执行所有 cell。**需要将文件路径改成 CUB-200-2011 数据集的实际存放路径。**

### Task 2（VOC 检测）

打开 `homework_task_2.ipynb`，前两段代码需要写入相应的配置文件中：
- Faster R-CNN 配置文件：`configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py`
- YOLO V3 配置文件：`configs/yolo/yolov3_d53_mstrain-608_273e_voc0712.py`

之后便可正常进行训练和测试。

## 数据集和模型权重

按原作业要求，需要的数据集和训练好的模型权重存放在 Google Drive：

> 📁 **Google Drive**：https://drive.google.com/drive/folders/1a9kk6ByBky7-oOSAvnkLMkvkMqxrjA4c?usp=drive_link

请自行从 Drive 下载，**并修改代码中的数据集绝对路径**。

## 任务描述（原文）

详见 [`HOMEWORK_ORIGINAL.txt`](./HOMEWORK_ORIGINAL.txt) 和 [`README_SOURCE.md`](./README_SOURCE.md)。
