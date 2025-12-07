import os
import subprocess
import sys

# 数据集所在的绝对路径
images_path = r"D:\yuzhm01\Desktop\nerf-windows\data\ggbond\input"

if not os.path.isdir(images_path):
    print(f"错误：未找到图片文件夹: {images_path}")
    sys.exit(1)

folder_path = os.path.dirname(images_path)
# 数据集名称文件夹 ggbond
dataset_name = os.path.basename(folder_path)

# COLMAP估算相机位姿
command = f'python convert.py -s {folder_path}'
subprocess.run(command, shell=True)

# 获取poses_bounds.npy脚本
command = f'python -m poses.pose_utils --datadir "{folder_path}" --match_type exhaustive_matcher'
subprocess.run(command, shell=True)

# 模型训练脚本，训练从folder_path/poses_bounds.npy中读取训练，，模型会保存在train_result\时间戳>\Iters\iter_count\Model 

# 快速管线自检（先跑通，看输出）：
# command = f'python run_nerf.py --datadir "{folder_path}" --factor 16 --iters 2000 --i_valid 200 --i_test 0'
# subprocess.run(command, shell=True)

# 标准训练（较完整，验证与测试间隔适中）：
command = f'python run_nerf.py --datadir "{folder_path}" --factor 8 --i_valid 1000 --i_test 2000 --iters 200000'
subprocess.run(command, shell=True)
