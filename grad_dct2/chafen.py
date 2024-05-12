import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def random_select_images(folder, count=100):
    """随机选择指定数量的图片文件"""
    files = os.listdir(folder)
    selected_files = random.sample(files, count)
    return [os.path.join(folder, file) for file in selected_files]

def compute_diff_array(image_path):
    """计算并返回图像的RGB通道二维差分数组"""
    image = Image.open(image_path)
    image_array = np.array(image)
    diff_arrays = [np.diff(image_array[:, :, channel], axis=0) for channel in range(3)]  # 对每个通道计算差分
    return diff_arrays

def visualize_and_save_diff_arrays(diff_arrays, save_folder, base_name):
    """将差分数组可视化并保存为图片"""
    for i, diff_array in enumerate(diff_arrays):
        plt.figure(figsize=(6, 6))
        plt.imshow(diff_array, cmap='viridis')
        plt.axis('off')
        plt.savefig(os.path.join(save_folder, f"{base_name}_channel_{i}.png"), bbox_inches='tight')
        plt.close()

def process_images(folder, save_folder):
    """处理指定文件夹中的图片，并保存差分数组的可视化结果"""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    selected_images = random_select_images(folder)
    for img_path in selected_images:
        base_name = os.path.basename(img_path).split('.')[0]
        diff_arrays = compute_diff_array(img_path)
        visualize_and_save_diff_arrays(diff_arrays, save_folder, base_name)

# 分别处理0_real和1_fake文件夹中的图片
process_images("/opt/data/private/wangjuntong/datasets/progan_train/car/0_real", "diff_real")
process_images("/opt/data/private/wangjuntong/datasets/progan_train/car/1_fake", "diff_fake")
