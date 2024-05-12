import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.fftpack import dct, idct
from scipy.signal import convolve2d
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 指定想要使用的GPU编号
gpu_number = 2

# 检查CUDA是否可用，然后指定设备
if torch.cuda.is_available():
    device = torch.device(f'cuda:{gpu_number}')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')



# DCT变换函数
def apply_dct(image, keep_ratio=0.1):
    image_np = np.array(image)
    dct_transformed = dct(dct(image_np, axis=0, norm='ortho'), axis=1, norm='ortho')
    h, w = dct_transformed.shape[:2]
    h_keep = int(h * keep_ratio)
    w_keep = int(w * keep_ratio)
    dct_transformed[h_keep:, :] = 0
    dct_transformed[:, w_keep:] = 0
    idct_transformed = idct(idct(dct_transformed, axis=0, norm='ortho'), axis=1, norm='ortho')
    idct_transformed_normalized = (idct_transformed - np.min(idct_transformed)) / (np.max(idct_transformed) - np.min(idct_transformed))
    image_uint8 = (idct_transformed_normalized * 255).astype(np.uint8)
    return Image.fromarray(image_uint8)

# 梯度变换函数
def generate_gradient_image(image):
    image_np = np.array(image)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad = np.zeros_like(image_np)
    for i in range(3):
        grad_x = np.abs(convolve2d(image_np[:, :, i], sobel_x, mode='same', boundary='wrap'))
        grad_y = np.abs(convolve2d(image_np[:, :, i], sobel_y, mode='same', boundary='wrap'))
        grad[:, :, i] = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad = np.max(grad, axis=2)
    gradient_normalized = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    gradient_uint8 = (gradient_normalized * 255).astype(np.uint8)
    return Image.fromarray(gradient_uint8)

# MultiHeadSelfAttention类
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values):
        N, C, H, W = values.size()
        values = values.view(N, C, -1).transpose(1, 2)  # Reshape to (N, H*W, C)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, H*W, self.heads, self.head_dim)
        keys = self.keys(values)
        queries = self.queries(values)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, H*W, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out.view(N, C, H, W)  # Reshape back to (N, C, H, W)



# FeatureFusionModule类
class FeatureFusionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FeatureFusionModule, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, 2)  # 生成两个权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        weight = self.sigmoid(x)
        return x1 * weight[:, 0:1].unsqueeze(2).unsqueeze(3) + x2 * weight[:, 1:2].unsqueeze(2).unsqueeze(3)

# DualBranchNetwork类
class DualBranchNetwork(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DualBranchNetwork, self).__init__()
        # DCT分支
        self.dct_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MultiHeadSelfAttention(16, 2)
        )
        # 梯度分支 - 注意这里的输入通道数改为1
        self.gradient_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 修改这里，使其接受单通道输入
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MultiHeadSelfAttention(16, 2)
        )
        # 特征融合模块
        self.feature_fusion = FeatureFusionModule(32, reduction=16)  # 两个分支各16个通道
        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, dct_x, grad_x):
        dct_features = self.dct_branch(dct_x)
        grad_features = self.gradient_branch(grad_x)
        fused_features = self.feature_fusion(dct_features, grad_features)
        out = self.classifier(fused_features)
        return out


# ProGAN_Dataset类
class ProGAN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
        dct_image = apply_dct(image)
        grad_image = generate_gradient_image(image)
        if self.transform:
            dct_image = self.transform(dct_image)
            grad_image = self.transform(grad_image)
        label = self.labels[idx]
        return dct_image, grad_image, label

# 数据预处理
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

def train_model(model, train_loader, val_loader, device, epochs=25):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # Keep this definition here
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for dct_images, grad_images, labels in train_loader:
            dct_images, grad_images, labels = dct_images.to(device), grad_images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(dct_images, grad_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        # Pass criterion as an argument to validate_model
        validate_model(model, val_loader, device, criterion)

def validate_model(model, val_loader, device, criterion):  # Accept criterion as an argument
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for dct_images, grad_images, labels in val_loader:
            dct_images, grad_images, labels = dct_images.to(device), grad_images.to(device), labels.to(device)
            outputs = model(dct_images, grad_images)
            loss = criterion(outputs, labels)  # Use the passed criterion
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')


def main():
    train_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/progan_train', transform=transform)
    val_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/CNN_synth_testset', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = DualBranchNetwork(in_channels=3, num_classes=2).to(device)

    train_model(model, train_loader, val_loader, device, epochs=25)

if __name__ == "__main__":
    main()
