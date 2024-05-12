import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import numpy as np

# 检查CUDA是否可用，选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
def load_data(file_name):
    features = []
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            features.append([float(p.split(':')[1]) for p in parts[1:]])
    return features

class ImageDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float).to(device), torch.tensor(self.labels[idx], dtype=torch.long).to(device)

# 加载数据
real_features = load_data('real.txt')
fake_features = load_data('fake.txt')

labels = [1] * len(real_features) + [0] * len(fake_features)
features = real_features + fake_features

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(features[0]), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 将模型移至选择的设备
model = Net().to(device)

# k-fold验证设置
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

features = np.array(features)
labels = np.array(labels)

# 主训练循环
for fold, (train_ids, test_ids) in enumerate(kfold.split(features)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # 数据准备
    train_features, test_features = features[train_ids], features[test_ids]
    train_labels, test_labels = labels[train_ids], labels[test_ids]
    
    train_dataset = ImageDataset(train_features.tolist(), train_labels.tolist())
    test_dataset = ImageDataset(test_features.tolist(), test_labels.tolist())
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练过程
    for epoch in range(1000):  # 迭代次数
        model.train()
        running_loss = 0.0
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证过程
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # 打印每个epoch的损失和准确率
        print(f'Epoch {epoch+1}/{1000}, Loss: {running_loss/len(train_loader)}, Accuracy: {100.0 * correct / total}%')

    print('--------------------------------')
