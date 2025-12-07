import os
import sys
import subprocess

# 视频绝对路径
video_path = r"D:\yuzhm01\Desktop\nerf-windows\data\ggbond\ggbond.mp4"

if not os.path.isfile(video_path):
	print(f"错误：未找到视频文件: {video_path}")
	sys.exit(1)

# 切分帧数，每秒多少帧
fps = 2

# 获取当前工作路径
current_path = os.getcwd()

folder_path = os.path.dirname(video_path)
# 数据集保存路径 data/ggbond/input
images_path = os.path.join(folder_path, 'input')
os.makedirs(images_path, exist_ok=True)
# 数据集名称文件夹 ggbond
dataset_name = os.path.basename(folder_path)

ffmpeg_path = os.path.join(current_path, 'external', r'ffmpeg/bin/ffmpeg.exe')

# 视频切分脚本
command = f'{ffmpeg_path} -i {video_path} -qscale:v 1 -qmin 1 -vf fps={fps} {images_path}\\%04d.jpg'
subprocess.run(command, shell=True)

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