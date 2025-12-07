# NeRF on Windows

Windows下NeRF 训练（PyTorch），从视频数据集经 COLMAP 估计相机位姿（可选colmap处理后的图片数据集）、生成 LLFF 兼容的 `poses_bounds.npy`，再进行 NeRF 训练与渲染。项目在 `external/` 下内置了 COLMAP（CUDA/无 CUDA）、FFmpeg 与 ImageMagick，尽量减少 Windows 环境的安装复杂度。

## 概述
- 目标：用真实数据视频训练 NeRF，并渲染新视角。
- 流程：帧抽取（FFmpeg） → COLMAP 重建/去畸变 → 生成 LLFF 位姿 → NeRF 训练/渲染。
- 训练与渲染：可配置的 NeRF MLP，位置编码，粗/精两阶段采样，螺旋/球形相机路径。
- 输出：日志、权重、验证帧与渲染视频统一保存到 `train_result/`。

## 环境安装
1) 安装 Python 依赖（Torch/torchvision 请根据你的 CUDA/CPU 环境选择合适版本）：
```
# CUDA 11.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```
- COLMAP：默认使用 `external/COLMAP-3.8-windows-cuda/COLMAP.bat`，也提供 `external/COLMAP-3.7-windows-no-cuda/COLMAP.bat`。
- FFmpeg：项目内置路径 `external/ffmpeg/bin/ffmpeg.exe`。
- ImageMagick：`convert.py` 在 `--resize` 时引用 `external/ImageMagick-7.1.1-Q16-HDRI/magick.exe`。
- 若使用系统安装的 COLMAP，请将包含 `COLMAP.bat` 的目录加入 PATH，保证在 `cmd.exe` 中执行 `colmap.bat` 正常。

2) 下载依赖包

在 Release 中下载 `external.zip` 压缩包，并将其解压到项目根目录下。解压后，项目根目录应包含 `external` 文件夹，其中包含 `ffmpeg`、`colmap` 等工具。

## 数据准备
数据集建议放在 `data/<scene>/`，视频和图片数据集均可（二选一）：
- 视频路线：将视频文件放在 `data/<scene>/<scene>.mp4`。
- 图片路线：colmap处理后的图片数据集，将 JPG/PNG 放入 `data/<scene>/input/`。

`convert.py`（COLMAP 管线）：

- 依次执行特征提取、穷举匹配、建图、图像去畸变。
- 生成：
  - `data/<scene>/images/`（去畸变后的训练用图像）
  - `data/<scene>/sparse/0/`（COLMAP 输出：`cameras.bin`、`images.bin`、`points3D.bin`）
  - （可选）`--resize` 时生成 `images_2/`、`images_4/`、`images_8/`（调用 ImageMagick）。

`poses/pose_utils.py`（LLFF 位姿转换）：
- 读取 `sparse/0/*.bin`，转换到 LLFF 坐标系，输出 `data/<scene>/poses_bounds.npy`。
- 若有未匹配上的图片，脚本会列出名单；请删除 `sparse/`、`colmap_output.txt`、`database.db` 以及这些图片后重新运行。

示例结构（`data/ggbond/`）：
- `input/` → 原始输入图片（或由视频切分得到的帧）
- `images/` → 去畸变后的训练图像
- `sparse/0/` → COLMAP 相机/位姿/点云二进制
- `poses_bounds.npy` → 由 `poses/pose_utils.py` 生成
- 可选：`images_2/`、`images_4/`、`images_8/`、`images_16/`

## 快速开始训练

> 自定义处理数据集以及训练的pipeline脚本

从视频训练：
```
python nerf_train_video.py
```
脚本会切分帧、运行 COLMAP、转换位姿并启动训练。

从图片训练（可选）：
```
python nerf_train_images.py
```
## 设置参数训练

最小自检（先跑通看输出）：
```
python run_nerf.py --datadir "d:\\yuzhm01\\Desktop\\nerf-windows\\data\\ggbond" --factor 16 --iters 2000 --i_valid 200 --i_test 0
```

标准训练（较完整，验证与测试间隔适中）：
```
python run_nerf.py --datadir "d:\\yuzhm01\\Desktop\\nerf-windows\\data\\ggbond" --factor 8 --i_valid 1000 --i_test 2000 --iters 200000
```

仅渲染（已有训练结果）：
```
python run_nerf.py --datadir "d:\\yuzhm01\\Desktop\\nerf-windows\\data\\ggbond" --render_only True
```

## 训练参数说明（`run_nerf.py`）
- 关键参数：
  - `--datadir`：数据根目录，需包含 `poses_bounds.npy` 与 `images/`。
  - `--factor`：数据下采样倍率（影响加载/内存占用）。
  - `--iters`、`--N_rand`、`--lr`、`--lr_decay`：训练超参。
  - `--N_samples`（粗）与 `--N_importance`（精）：每条光线的采样数。
  - `--use_viewdirs`、`--multires`、`--multires_views`：位置编码相关配置。
  - 场景类型：`--no_ndc`、`--lindisp`、`--spherify`（360 场景通常置 True；前向场景保持 False）。
  - 数据划分：`--llffhold`（每隔 N 张取一张为测试，默认 8）。
  - 频率：`--i_log`、`--i_valid`、`--i_test` 控制日志/验证/渲染间隔。
- 备注：
  - 前向场景一般 `--no_ndc False`、`--lindisp False`、`--spherify False`；360 场景设为 True。
  - `--render_test True` 使用测试集位姿渲染而非螺旋路径。
  - 首次训练可将 `--i_valid` 调小（如 100/200）快速自检。

## 输出与预览
训练会在 `train_result/<YYYY_MM_DD_hh_mm_ss>/` 下生成：
- `Log/` → 超参（`args.txt`）、TensorBoard 日志、loss 图/文本。
- `Iters/<iter>/Model/` → 权重 `model-iter{iter}.pth`、`modelfine-iter{iter}.pth`。
- `Iters/<iter>/Valid_Result/` → 验证渲染 PNG。
- `Iters/<iter>/Render_Result/` → 螺旋路径渲染 PNG 与 MP4（如 `spiral_{iter}_rgb.mp4`、`spiral_{iter}_disp.mp4`）。
- 仅渲染模式：`render_only_result/video.mp4`。

## 脚本说明
- `nerf_train_images.py`：对 `data/<scene>/` 执行 `convert.py` → `python -m poses.pose_utils` → `run_nerf.py`。
- `nerf_train_video.py`：用 `external/ffmpeg/bin/ffmpeg.exe` 切帧，再执行同样的流程。
- `convert.py`：COLMAP 特征、匹配、建图、去畸变；可选 ImageMagick 批量缩放。
- `poses/pose_utils.py`：读取 COLMAP 输出并写出 LLFF `poses_bounds.npy`。
- `run_nerf.py`：训练/渲染主循环；负责日志、权重与渲染结果保存。
- `nerf_model.py`、`run_nerf_helpers.py`、`load_real_data.py`：模型、渲染工具与 LLFF 加载。

## 常见问题（FAQ）
- 图片未匹配：`poses/pose_utils.py` 会给出未匹配名单；删除 `sparse/`、`colmap_output.txt`、`database.db` 以及这些图片后重跑 `convert.py`。
- `imageio` 版本：若高版本报错，建议用 `2.9.0`（当前未在 `requirements.txt` 固定）。
- 显存不足：调小 `--N_rand`、`--netwidth`；或增大 `--factor` 以降低分辨率。
- 场景类型：根据前向/360 设置 `--no_ndc`/`--lindisp`/`--spherify`。

## 项目结构
```
convert.py
nerf_model.py
run_nerf_helpers.py
run_nerf.py
nerf_train_images.py
nerf_train_video.py
load_real_data.py
poses/
  pose_utils.py
  colmap_wrapper.py
  colmap_read_model.py
external/
  COLMAP-3.8-windows-cuda/
  COLMAP-3.7-windows-no-cuda/
  ffmpeg/
  ImageMagick-7.1.1-Q16-HDRI/
train_result/
data/<scene>/
```

## 参考
  - NeRF 论文：[NeRF: Representing Scenes as
Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
  - nerf-pytorch参考 [yenchenlin/nerf-pytorch](https://github.com/yzm-dev/nerf-pytorch/tree/master) 实现。
  - COLMAP：[colmap](https://colmap.github.io/)
  - LLFF 位姿工具源自 [Fyusion/LLFF](https://github.com/Fyusion/LLFF/tree/master/llff/poses)。