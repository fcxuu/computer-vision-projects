# CIFAR-100 自监督 vs 监督（期末作业 Task 1）

**来源仓库**：`cxk049/final_project`（已归档）

## 项目简介

计算机视觉课程期末作业 Task 1 —— 对比监督学习和自监督学习在图像分类任务上的性能表现。

- 任选一种自监督学习算法训练 ResNet-18
- 在 CIFAR-100 上用 Linear Classification Protocol 评测
- 与 ImageNet 监督预训练、从零训练三组对比

## 文件结构

```
cifar100-self-vs-supervised/
├── homework_task_1.ipynb       # 主 Notebook
├── HOMEWORK_ORIGINAL.txt       # 原任务描述（保留原文，含 Task 1/2/3）
├── README_SOURCE.md            # 原仓库 README 备份
└── README.md                   # 本文件
```

## 使用说明

打开 `homework_task_1.ipynb`，按顺序执行所有 cell。

数据准备：
- 自监督预训练的数据集自选
- Linear Classification 在 CIFAR-100 上做
- 需要 ImageNet 监督预训练权重做对比

## 任务描述（原文）

详见 [`HOMEWORK_ORIGINAL.txt`](./HOMEWORK_ORIGINAL.txt)（含 Task 1 / 2 / 3 全部描述）和 [`README_SOURCE.md`](./README_SOURCE.md)。
