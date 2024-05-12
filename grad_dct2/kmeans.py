import pandas as pd
from sklearn.cluster import KMeans

# 假设txt文件内容已经按照提供的格式保存，每行一个记录
# 首先，我们将从txt文件中读取数据

# 定义读取数据的函数
def read_data_from_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(': ')
            image_name = parts[0]
            features = parts[1].split(', ')
            features = [float(feature.split('=')[1]) for feature in features]
            data.append((image_name, *features))
    return data

# 假设文件路径为"/mnt/data/huidu.txt"，这里读取数据
data = read_data_from_txt("huidu.txt")

# 将数据加载到Pandas DataFrame中
df = pd.DataFrame(data, columns=["Image", "ASM", "Contrast", "Entropy", "IDM"])

# 使用K-means算法进行聚类
kmeans = KMeans(n_clusters=2, random_state=0).fit(df[["ASM", "Contrast", "Entropy", "IDM"]])

# 将聚类结果添加到DataFrame中
df['Cluster'] = kmeans.labels_

def write_cluster_results_to_txt(df, file_path):
    with open(file_path, 'w') as file:
        for index, row in df.iterrows():
            line = f"{row['Image']}: Cluster={row['Cluster']}\n"
            file.write(line)

# 定义输出文件路径
output_file_path = "cluster_output.txt"

# 调用函数，将模拟的聚类结果写入到txt文件
write_cluster_results_to_txt(df, output_file_path)
