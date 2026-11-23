# Buddha 多视角实拍 NeRF 重建

本实验在 Apple M1 Pro 上运行。PyCOLMAP 由实拍 RGB 照片恢复相机位姿并去畸变。NeRF 复用固定版本框架的场网络、分层采样和体积渲染，并使用每张图的 COLMAP 内参生成光线。输入照片来自公开的 Buddha Dataset，保留原作业和原有文件。

## 运行

从仓库根目录执行。依赖为 Python 3.12、FFmpeg 和 ffprobe。COLMAP 和 PyTorch 各在独立的 Python 进程中运行。

```bash
project_dir=03-self-supervised-and-transformer/nerf-object-reconstruction/reproduction
uv venv --python 3.12
uv pip install -r "$project_dir/requirements.txt"
mkdir -p data/nerf-buddha data/vendor runs

curl -fL --retry 3 -C - -o data/nerf-buddha/dataset_buddha-master.zip https://zenodo.org/api/records/1203283/files/dataset_buddha-master.zip/content
git clone https://github.com/sadeeqgandalf/Nerf-3D-Reconstruction.git data/vendor/nerf
git -C data/vendor/nerf checkout --detach 9c47e83c5a2c9736c9e770cc993237ce3e2667ed

.venv/bin/python "$project_dir/prepare.py" --archive data/nerf-buddha/dataset_buddha-master.zip --out data/nerf-buddha/scene
.venv/bin/python "$project_dir/train.py" --scene data/nerf-buddha/scene/scene.json --device mps --out runs/buddha-nerf
.venv/bin/python "$project_dir/train.py" --scene data/nerf-buddha/scene/scene.json --device mps --out runs/buddha-nerf-focused --initialize-from runs/buddha-nerf/best.pt --focus-share 0.6 --focus-radius 1.1 --lr 0.00025 --steps 3500 --validate-every 700
.venv/bin/python "$project_dir/outputs.py" --scene data/nerf-buddha/scene/scene.json --vendor data/vendor/nerf --device mps test --checkpoint runs/buddha-nerf-focused/best.pt --out runs/buddha-results --trajectory runs/buddha-nerf-focused/checkpoints/*.pt
.venv/bin/python "$project_dir/outputs.py" --scene data/nerf-buddha/scene/scene.json --vendor data/vendor/nerf --device mps orbit --checkpoint runs/buddha-nerf-focused/best.pt --out runs/buddha-results
.venv/bin/python "$project_dir/report.py" --scene data/nerf-buddha/scene/scene.json --run runs/buddha-nerf-focused --other-run runs/buddha-nerf --out runs/buddha-results
```

回载已发布模型时，下载并解压同一 Release 中的 `buddha-scene.zip`，用其中的 `scene.json` 和图片作为测试输入，将 `--checkpoint` 改为 `buddha-nerf.pt`。压缩包中保留的是实际训练用的去畸变 RGB、标定配置及 COLMAP 稀疏模型。CPU 模式可将 `--device` 改为 `cpu`。

## 实验协议

- 对原作者归档校验 MD5，只提取 67 张完整序列实拍照片。CPU SIFT 穷举匹配、增量建图，使用单相机模型和估计内参去畸变。
- 所有注册照片参与 COLMAP 几何标定。NeRF 的训练、验证及测试按图像编号固定交错留出。只有训练图的 RGB 参与优化。每隔固定步数渲染完整验证图，以平均 PSNR 选择权重。先跑 7,000 步均匀光线基线，再从基线最佳轮微调 3,500 步。微调中 60% 的抽样来自穿过场景中心半径 1.1 球体的光线，其余均匀抽取。两阶段共用全图验证协议。
- 从训练图均匀抽取像素光线，在粗、细两级上优化 RGB 均方误差之和。独立测试命令冻结权重并重新渲染测试视角，逐图报告 PSNR、SSIM、MSE。测试轨迹在验证选型完成后离线回放。
- 环绕路径根据实拍相机的典型距离和俯仰估计，共 90 张新视图。路径的世界上方向依据相机中心与四个候选图像轴估计，兼顾原序列混合的横竖拍。视频帧数、分辨率和帧率由 ffprobe 核验。

原始照片由 Simone Gasparini、Fabien Castan、Yann Lanthony 提供，发布于 2017 年。测试对照和环绕视频从其素材派生，经过缩放、去畸变和 NeRF 渲染。[Buddha Dataset](https://doi.org/10.5281/zenodo.1203283) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [NeRF 框架及 MIT 许可证](https://github.com/sadeeqgandalf/Nerf-3D-Reconstruction/tree/9c47e83c5a2c9736c9e770cc993237ce3e2667ed)。

## 实测结果

COLMAP 注册 **67 / 67** 张实拍照片，重建 **6,742** 个稀疏点，平均重投影误差 **0.341 像素**。训练、验证、测试视角分别为 **50 / 8 / 9**。共用同一套标定和留出划分。

| 检查点 | 最佳验证 PSNR | 测试 PSNR | 测试 SSIM | 测试 MSE |
|---|---:|---:|---:|---:|
| 均匀采样，第 5,000 步 | 17.57 dB | 17.36 dB | 0.4666 | 0.021775 |
| 加密雕像附近光线，第 2,100 步微调 | **17.63 dB** | **17.57 dB** | **0.4819** | **0.021125** |

按验证集结果选中微调检查点。全图测试 PSNR 较均匀采样高约 0.21 dB。渲染保留了佛头像的轮廓和整体形状，近景纹理仍偏软，部分背景有条纹。环绕相机使用交错拍摄校正后的上方向，并放宽焦距对应的视场以保留完整雕像。

- [PDF 报告](evidence/report.pdf) · [训练与留出曲线](evidence/training-curves.png) · [未参与训练的视角对比](evidence/test-comparison.png)
- [相机参数与划分](evidence/scene.json) · [环绕预览](evidence/orbit-preview.png) · [视频编码记录](evidence/orbit-ffprobe.json) · [光线坐标验算](evidence/verification.txt)
- [准备好的场景、原始对照权重、最佳权重与环绕视频](https://github.com/fcxuu/computer-vision-projects/releases/tag/portfolio-nerf-buddha-2026-11-24) · [SHA-256](evidence/SHA256SUMS)
