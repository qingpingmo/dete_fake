import matplotlib.pyplot as plt
import numpy as np

# 定义一个函数来读取数据
def read_data(filepath):
    data = {
        'ASM': [],
        'Contrast': [],
        'Entropy': [],
        'IDM': []
    }
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.split(': ')
            if len(parts) == 2:
                features = parts[1].split(', ')
                data['ASM'].append(float(features[0].split('=')[1]))
                data['Contrast'].append(float(features[1].split('=')[1]))
                data['Entropy'].append(float(features[2].split('=')[1]))
                data['IDM'].append(float(features[3].split('=')[1]))
    return data

# 读取数据
data = read_data('huidu.txt')

# 绘制图表的函数
def plot_3d(x, y, z, x_label, y_label, z_label, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    plt.savefig(filename)
    plt.close()

# 绘制图表
plot_3d(data['ASM'], data['Contrast'], data['Entropy'], 'ASM', 'Contrast', 'Entropy', 'ASM vs Contrast vs Entropy', 'ASM_Contrast_Entropy.png')
plot_3d(data['ASM'], data['IDM'], data['Entropy'], 'ASM', 'IDM', 'Entropy', 'ASM vs IDM vs Entropy', 'ASM_IDM_Entropy.png')
plot_3d(data['Contrast'], data['IDM'], data['Entropy'], 'Contrast', 'IDM', 'Entropy', 'Contrast vs IDM vs Entropy', 'Contrast_IDM_Entropy.png')

print("绘图完成，图表已保存。")
