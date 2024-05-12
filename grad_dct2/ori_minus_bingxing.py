
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import numpy as np
from scipy.fftpack import dct, idct
import torch.nn as nn
from math import sqrt
from scipy.signal import convolve2d
from sklearn.metrics import average_precision_score
from torchvision.models import resnet
import io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import random
import torch
import random
from PIL import ImageFilter, ImageOps,UnidentifiedImageError
import PIL


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        kernel_size = kernel_size if kernel_size is not None else 3
        stride = stride if stride is not None else 1
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
            for _ in range(num_capsules)
        ])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            priors = torch.cat(priors, dim=-1)
            logits = torch.zeros(*priors.size()).to(x.device)

            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i < self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits

        else:
            outputs = [capsule(x) for capsule in self.capsules]
            outputs = torch.stack(outputs, dim=1)
            outputs = outputs.view(x.size(0), -1, outputs.size(-1))
            outputs = self.squash(outputs)

        return outputs


def apply_dct(image):
    image_np = np.array(image)
    dct_transformed = dct(dct(image_np, axis=0, norm='ortho'), axis=1, norm='ortho')
    # 将 DCT 变换的结果转换为图像格式，保持大小不变
    dct_image = np.log(np.abs(dct_transformed) + 1)  # 使用对数尺度以便更好地可视化
    dct_image_normalized = (dct_image - np.min(dct_image)) / (np.max(dct_image) - np.min(dct_image))
    image_uint8 = (dct_image_normalized * 255).astype(np.uint8)
    return Image.fromarray(image_uint8)

def generate_gradient_image(image):
    image_np = np.array(image)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad = np.zeros_like(image_np)
    for i in range(3):  # 对每个颜色通道分别应用
        grad_x = np.abs(convolve2d(image_np[:, :, i], sobel_x, mode='same', boundary='wrap'))
        grad_y = np.abs(convolve2d(image_np[:, :, i], sobel_y, mode='same', boundary='wrap'))
        grad[:, :, i] = np.sqrt(grad_x ** 2 + grad_y ** 2)

    grad = np.max(grad, axis=2) 
    gradient_normalized = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    gradient_uint8 = (gradient_normalized * 255).astype(np.uint8)

    return Image.fromarray(gradient_uint8)


class ProGAN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, feature_dir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.feature_dir = feature_dir if feature_dir is not None else os.path.join(self.root_dir, 'new_features')
        os.makedirs(self.feature_dir, exist_ok=True)
        self.image_paths = []
        self.labels = []
        self._load_images(self.root_dir)

    def _load_images(self, current_dir):
        for item in os.listdir(current_dir):  # 遍历当前目录
            path = os.path.join(current_dir, item)  # 获取文件或目录的完整路径
            if os.path.isdir(path):  # 如果是目录，递归调用加载图像
                self._load_images(path)
            elif path.endswith('.png'):  # 如果是PNG图像文件
                label = 1 if '1_fake' in path else 0  # 根据文件名判断图像是否为假图像，并设置标签
                self.image_paths.append(path)  # 添加图像路径到列表
                self.labels.append(label)  # 添加标签到列表

    def __len__(self):
        return len(self.image_paths)  # 返回数据集中图像的数量

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        feature_path = self._get_feature_path(img_path)
        if os.path.exists(feature_path):
            try:
                new_feature_image = Image.open(feature_path).convert('RGB')
            except PIL.UnidentifiedImageError as e:
                print(f"Error opening image {feature_path}: {e}")
                # Optionally, process the image again if the existing one is invalid
                new_feature_image = self._process_image(image)
                new_feature_image.save(feature_path)
        else:
            new_feature_image = self._process_image(image)
            new_feature_image.save(feature_path)

        if self.transform:
            image = self.transform(image)
            new_feature_image = self.transform(new_feature_image)

        label = self.labels[idx]
        return image, new_feature_image, label



    def _process_image(self, image):
        image_array = np.array(image)
        new_feature_image_array = np.zeros_like(image_array)
        
        for channel in range(3):  # 对每个颜色通道
            channel_data = image_array[:,:,channel]
            for i in range(0, channel_data.shape[0], 2):
                for j in range(0, channel_data.shape[1], 2):
                    block = channel_data[i:i+2, j:j+2]
                    block_mean = np.mean(block, dtype=np.float64)
                    new_feature_image_array[i:i+2, j:j+2, channel] = block - block_mean
        
        new_feature_image = Image.fromarray(new_feature_image_array.clip(0, 255).astype(np.uint8))
        return new_feature_image
    
    def _get_feature_path(self, img_path):
        # Generate a unique file path for the new feature image based on the original image path
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        feature_name = f"{name}_new_feature{ext}"
        feature_path = os.path.join(self.feature_dir, feature_name)
        return feature_path


transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5]) 
])


progan_dataset = ProGAN_Dataset(root_dir='/opt/data/private/wangjuntong/datasets/progan_train', transform=transform)
train_size = int(0.8 * len(progan_dataset))
val_size = len(progan_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(progan_dataset, [train_size, val_size])
batch_size = 16
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

import torch.nn as nn
import torch.nn.functional as F


class FusionCapsNet(nn.Module):
    def __init__(self):
        super(FusionCapsNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            # Adjusted to take a single image as input
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.cbam = CBAM(in_planes=512) 
        self.capsule_layer = CapsuleLayer(
            num_capsules=10,
            num_route_nodes=-1,
            in_channels=512, 
            out_channels=32,
            kernel_size=None,
            stride=None
        )

        self.classifier = nn.Linear(288000, 2) 

    def forward(self, x, x_new_feature):
        features = self.feature_extractor(x)
        new_feature = self.feature_extractor(x_new_feature)
        combined_features = torch.cat([features, new_feature], dim=1)
        combined_features = self.cbam(combined_features)
        capsule_output = self.capsule_layer(combined_features)
        capsule_output = capsule_output.view(x.size(0), -1)
        outputs = self.classifier(capsule_output)
        return outputs

def ResNet18():
    return FusionCapsNet()


net = ResNet18()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

net.to(device)
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def train(trainloader, model, criterion, optimizer, epoch, scheduler, warm):
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0  # 累计损失
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数

    for i, (images, new_feature_images, labels) in enumerate(trainloader):
        images, new_feature_images, labels = images.to(device), new_feature_images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, new_feature_images)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 累加损失
        _, predicted = outputs.max(1)  # 获取预测的类别
        total += labels.size(0)  # 更新总样本数
        correct += predicted.eq(labels).sum().item()  # 更新正确预测的数量

        if epoch < warm:  # 如果处于预热阶段
            scheduler.step()  # 更新学习率

    train_loss = running_loss / len(trainloader)  # 计算平均训练损失
    acc = 100. * correct / total  # 计算训练准确率
    return train_loss, acc  # 返回训练损失和准确率

# 定义validate函数，用于验证模型
def validate(val_loader, model, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []

    with torch.no_grad():  # Disabling gradient calculation
        for i, (images, new_feature_images, labels) in enumerate(val_loader):
            images, new_feature_images, labels = images.to(device), new_feature_images.to(device), labels.to(device)
            outputs = model(images, new_feature_images)  # Adjusted to take two inputs
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    acc = 100 * correct / total
    ap = average_precision_score(all_targets, all_predicted)  # Compute average precision score

    return val_loss, acc, ap






warm = 1
epoch = 160
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

def main():
    # 创建训练数据集实例
    train_dataset = ProGAN_Dataset(root_dir='/opt/data/private/wangjuntong/datasets/progan_train', transform=transform)
    # 创建训练数据加载器
    trainloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=5, pin_memory=True)
    # 实例化模型并移动到指定的设备
    net = ResNet18().to(device)
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-2)
    # 定义学习率调度器
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    # 创建学习率预热调度器
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
    # 初始化最佳准确率
    best_acc = 0
    # 开始训练周期
    for epoch in range(500):
        # 训练模型
        train_loss, train_acc = train(trainloader, net, loss_function, optimizer, epoch, warmup_scheduler, warm)
        # 如果结束预热，更新学习率
        if epoch >= warm:
            train_scheduler.step()

        # 打印训练结果
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}')

        if epoch > 25:
            folders = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
            # 遍历每个文件夹进行验证
            for folder in folders:
                print(f"Validating on {folder}...")
                # 构建验证数据集路径
                val_dataset_folder = os.path.join('/opt/data/private/wangjuntong/code/CNN_synth_testset', folder)
                # 创建验证数据集实例
                val_dataset = ProGAN_Dataset(root_dir=val_dataset_folder, transform=transform)
                # 创建验证数据加载器
                valloader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=5, pin_memory=True)

                # 验证模型
                val_loss, val_acc, val_ap = validate(valloader, net, loss_function)
                # 打印验证结果
                print(f'Folder: {folder}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val AP: {val_ap:.3f}')

                # 更新最佳准确率
                if val_acc > best_acc:
                    best_acc = val_acc
        # 打印最佳验证准确率
    print(f'Best Validation Accuracy: {best_acc:.3f}%')

# 程序入口
if __name__ == "__main__":
    main()  # 执行main函数