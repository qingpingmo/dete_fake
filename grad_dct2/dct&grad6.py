import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from scipy.fftpack import dct, idct
from scipy.signal import convolve2d
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 特征提取分支
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=(8, 8)):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.adaptive_pool(x)  # 确保输出特征图的尺寸为(output_size[0], output_size[1])
        return x


# 融合策略预测分支
class FusionStrategy(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionStrategy, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.sigmoid(x)

# 特征融合操作
class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        self.strategy = FusionStrategy(channels, 1)  # Outputs a single weight

    def forward(self, dct_features, gradient_features):
        # Compute fusion weights
        fusion_weights = self.strategy(dct_features + gradient_features)
        # Adjust the shape of fusion_weights to match the shape of dct_features
        fusion_weights = fusion_weights.view(-1, 1, 1, 1).expand_as(dct_features)
        # Ensure dimension compatibility for element-wise operations
        if dct_features.shape != gradient_features.shape:
            raise ValueError(f"Dimension mismatch between dct_features {dct_features.shape} and gradient_features {gradient_features.shape}")
        # Apply fusion weights
        return fusion_weights * dct_features + (1 - fusion_weights) * gradient_features



# 分类网络
class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)

# 完整的自适应融合网络
class AdaptiveFusionNet(nn.Module):
    def __init__(self, num_classes=2, feature_size=(8, 8)):
        super(AdaptiveFusionNet, self).__init__()
        self.dct_extractor = FeatureExtractor(3, 64, output_size=feature_size)
        self.gradient_extractor = FeatureExtractor(3, 64, output_size=feature_size)
        self.fusion = FeatureFusion(64)
        self.classifier = Classifier(64 * feature_size[0] * feature_size[1], num_classes)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected input to be a 4D tensor, got {x.dim()}D")
        if x.shape[1] != 3:
            raise ValueError(f"Expected each sample to have 3 channels, got {x.shape[1]}")
        dct_features = self.dct_extractor(apply_dct(x))
        gradient_features = self.gradient_extractor(generate_gradient_image(x))
        fused_features = self.fusion(dct_features, gradient_features)
        fused_features = torch.flatten(fused_features, 1)
        out = self.classifier(fused_features)
        return out



# DCT变换函数
def apply_dct(image_tensor, keep_ratio=0.1):
    # 转换前确保Tensor在CPU上，并转换为numpy数组
    image_np = image_tensor.cpu().numpy()
    result_images = []

    for img in image_np:  # 遍历批次中的每张图像
        dct_channels = []
        for channel in img:  # 遍历每个通道
            channel_dct = dct(dct(channel.T, norm='ortho').T, norm='ortho')
            h, w = channel_dct.shape
            h_keep = int(h * keep_ratio)
            w_keep = int(w * keep_ratio)
            channel_dct[h_keep:, :] = 0
            channel_dct[:, w_keep:] = 0
            channel_idct = idct(idct(channel_dct.T, norm='ortho').T, norm='ortho')
            dct_channels.append(channel_idct)
        
        dct_image = np.stack(dct_channels, axis=0)  # 将通道重新堆叠为一张图像
        result_images.append(dct_image)

    result_tensors = np.stack(result_images, axis=0)
    result_tensors = torch.from_numpy(result_tensors).float().to(device)
    result_tensors = F.interpolate(result_tensors, size=(8, 8), mode='bilinear', align_corners=False)
    return result_tensors

def generate_gradient_image(image_tensor):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    image_np = image_tensor.cpu().numpy()
    result_images = []

    for img in image_np:  # 遍历批次中的每张图像
        grad_channels = []
        for channel in img:  # 遍历每个通道
            grad_x = np.abs(convolve2d(channel, sobel_x, mode='same', boundary='wrap'))
            grad_y = np.abs(convolve2d(channel, sobel_y, mode='same', boundary='wrap'))
            grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_channels.append(grad)
        
        grad_image = np.stack(grad_channels, axis=0)
        result_images.append(grad_image)

    result_tensors = np.stack(result_images, axis=0)
    result_tensors = torch.from_numpy(result_tensors).float().to(device)
    result_tensors = F.interpolate(result_tensors, size=(8, 8), mode='bilinear', align_corners=False)
    return result_tensors

    



# 数据集准备和训练逻辑
class ProGAN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform  # 添加一个变换属性
        self.image_paths = []
        self.labels = []
        self._load_images(self.root_dir)

    def _load_images(self, current_dir):
        for item in os.listdir(current_dir):
            path = os.path.join(current_dir, item)
            if os.path.isdir(path):
                self._load_images(path)
            elif path.endswith('.png'):
                label = 1 if '1_fake' in path else 0
                self.image_paths.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)  # 应用变换
        return image, label



transform = Compose([
    Resize((128, 128)),  # 保持这个尺寸，如果你的模型需要不同的输入尺寸，这里也需要做相应的改变
    ToTensor(),
])

train_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/progan_train', transform=transform)
val_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/CNN_synth_testset', transform=transform)


# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 初始化模型、损失函数、优化器
model = AdaptiveFusionNet(num_classes=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练和验证
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, loss_fn, device)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')
