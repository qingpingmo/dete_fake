import csv

# 定义一个函数来处理转换逻辑
def convert_txt_to_csv(txt_file, csv_file):
    # 打开txt文件和准备好csv文件
    with open(txt_file, 'r') as txt, open(csv_file, 'w', newline='') as csvfile:
        # CSV文件的写入器
        csvwriter = csv.writer(csvfile)
        # 写入CSV文件的表头
        csvwriter.writerow(['Image', 'Energy', 'Entropy', 'InverseDifference', 'Correlation', 'InfoMeasureCorrelation1', 'InfoMeasureCorrelation2', 'Contrast', 'DifferenceEntropy', 'DifferenceVariance', 'SumAverage', 'SumEntropy', 'SumVariance'])
        
        # 遍历txt文件的每一行
        for line in txt:
            # 移除行尾的换行符，并以逗号分隔
            data = line.strip().split(',')
            # 获取图片名称
            image_name = data[0]
            # 除去图片名称后，剩下的是特征值，这里通过列表推导式处理每个特征值，移除前缀，只保留数值部分
            features = [item.split(':')[1] for item in data[1:]]
            # 将图片名称和特征值写入CSV
            csvwriter.writerow([image_name] + features)

# 转换真图片的特征值
convert_txt_to_csv('real.txt', 'real.csv')

# 转换假图片的特征值
convert_txt_to_csv('fake.txt', 'fake.csv')
