import os
import random
from PIL import Image
import numpy as np

# 创建保存图片的新文件夹
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 处理图片的函数
def process_image(image_path, output_dir):
    # 加载图片
    image = Image.open(image_path)
    image_array = np.array(image)

    # 对每个通道进行处理
    for channel in range(3):  # 对于RGB的每个通道
        # 提取单个通道的数据
        channel_data = image_array[:,:,channel]

        # 创建一个新的空数组，用于存储处理后的通道数据
        new_channel_data = np.zeros_like(channel_data)

        # 遍历图片的每个2x2区块
        for i in range(0, channel_data.shape[0], 2):
            for j in range(0, channel_data.shape[1], 2):
                # 提取2x2区块
                block = channel_data[i:i+2, j:j+2]

                # 计算区块的平均值
                block_mean = np.mean(block, dtype=np.float64)

                # 更新区块数据
                new_channel_data[i:i+2, j:j+2] = block - block_mean

        # 保存处理后的通道数据为图片
        Image.fromarray(new_channel_data.clip(0, 255).astype(np.uint8)).save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_channel{channel}.png"))

# 随机选择并处理图片的函数
def process_random_images(source_dir, output_dir, num_images=100):
    # 确保输出目录存在
    create_dir_if_not_exists(output_dir)

    # 获取所有图片文件名
    all_images = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.png')]
    # 随机选择指定数量的图片
    selected_images = random.sample(all_images, num_images)

    # 对每张选中的图片进行处理
    for image_path in selected_images:
        process_image(image_path, output_dir)

# 主执行函数
def main():
    real_images_dir = '/opt/data/private/wangjuntong/datasets/progan_train/car/0_real'
    fake_images_dir = '/opt/data/private/wangjuntong/datasets/progan_train/car/1_fake'
    output_real_dir = 'processed_images/real'
    output_fake_dir = 'processed_images/fake'

    # 分别处理real和fake文件夹中的图片
    process_random_images(real_images_dir, output_real_dir)
    process_random_images(fake_images_dir, output_fake_dir)

    print("图片处理完成。")

if __name__ == "__main__":
    main()
