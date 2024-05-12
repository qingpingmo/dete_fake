# 引入必要的包
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import efficientnet_b0, resnext50_32x4d
from torchvision.models import ResNeXt50_32X4D_Weights
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, resnext50_32x4d, EfficientNet_B0_Weights, ResNet50_Weights
import numpy as np
from PIL import Image
import os
from scipy.fftpack import dct, idct
from scipy.signal import convolve2d
import torch.nn.functional as F



# 指定只使用GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# 对于单通道图像的预处理操作
transform_single_channel = Compose([
    Resize((224, 224)),  # 或保持您的目标大小不变
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])  # 适用于单通道图像
])

# 对于三通道图像的预处理操作
transform_three_channel = Compose([
    Resize((224, 224)),  # 或保持您的目标大小不变
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的标准化参数，适用于三通道图像
])


# EfficientNet和ResNeXt模型的定义
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetModel, self).__init__()
        # 使用weights参数
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.efficientnet = efficientnet_b0(weights=weights)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_classes)

class ResNeXtModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNeXtModel, self).__init__()
        # 使用weights参数
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        self.resnext = resnext50_32x4d(weights=weights)
        num_ftrs = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(num_ftrs, num_classes)



class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        # 如果kernel_size或stride为None，则分别使用默认值3和1
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



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out



class AttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def apply_dct(image, keep_ratio=0.1):
    image_np = np.array(image)
    # 应用二维DCT
    dct_transformed = dct(dct(image_np, axis=0, norm='ortho'), axis=1, norm='ortho')

    # 保留一定比例的DCT系数
    h, w = dct_transformed.shape[:2]
    h_keep = int(h * keep_ratio)
    w_keep = int(w * keep_ratio)
    dct_transformed[h_keep:, :] = 0
    dct_transformed[:, w_keep:] = 0

    # 应用逆DCT变换
    idct_transformed = idct(idct(dct_transformed, axis=0, norm='ortho'), axis=1, norm='ortho')

    # 标准化到0-255并转换为uint8
    idct_transformed_normalized = (idct_transformed - np.min(idct_transformed)) / (np.max(idct_transformed) - np.min(idct_transformed))
    image_uint8 = (idct_transformed_normalized * 255).astype(np.uint8)

    return Image.fromarray(image_uint8)

def generate_gradient_image(image):
    """
    生成图像的梯度图像。
    """
    image_np = np.array(image)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad = np.zeros_like(image_np)
    for i in range(3):  # 对每个颜色通道分别应用
        grad_x = np.abs(convolve2d(image_np[:, :, i], sobel_x, mode='same', boundary='wrap'))
        grad_y = np.abs(convolve2d(image_np[:, :, i], sobel_y, mode='same', boundary='wrap'))
        grad[:, :, i] = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 标准化到0-255并转换为uint8
    grad = np.max(grad, axis=2)  # 将所有通道合并为一个
    gradient_normalized = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    gradient_uint8 = (gradient_normalized * 255).astype(np.uint8)

    return Image.fromarray(gradient_uint8)


# ProGAN_Dataset类定义
class ProGAN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform_three_channel = transform_three_channel
        self.transform_single_channel = transform_single_channel
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
    
    # 应用DCT变换并使用三通道图像的转换
        dct_image = apply_dct(image)
        dct_image = self.transform_three_channel(dct_image)

        # 应用梯度变换并使用单通道图像的转换
        grad_image = generate_gradient_image(image)
        grad_image = self.transform_single_channel(grad_image)

        label = self.labels[idx]
        return dct_image, grad_image, label




# 数据集的预处理操作
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])  # 适用于单通道图像
])


# 实例化ProGAN_Dataset
progan_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/progan_train', transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(progan_dataset))
val_size = len(progan_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(progan_dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 16
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

import torch.nn as nn
import torch.nn.functional as F

# BasicBlock类
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_attention=False, heads=4):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = MultiHeadSelfAttention(planes, heads)

        

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_attention:
            out = out.view(out.shape[0], -1, out.shape[1]) # 调整形状以匹配自注意力层输入
            out = self.attention(out, out, out, None)
            out = out.view(out.shape[0], out.shape[2], int(sqrt(out.shape[1])), int(sqrt(out.shape[1]))) # 恢复形状


        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet类
class FusionCapsNet(nn.Module):
    def __init__(self):
        super(FusionCapsNet, self).__init__()
        
        # DCT特征提取部分
        self.dct_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        
        # 梯度特征提取部分
        self.grad_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        # Capsule层
       # Capsule层
        self.capsule_layer = CapsuleLayer(
            num_capsules=10, 
            num_route_nodes=-1, 
            in_channels=256,  # 更新为与融合特征通道数匹配
            out_channels=32, 
            kernel_size=None, 
            stride=None
        )
        
        # 分类器
        self.classifier = nn.Linear(933120, 2)  # 调整输入特征数量

    def forward(self, x_dct, x_grad):
        # DCT特征提取
        dct_features = self.dct_feature_extractor(x_dct)
        #print("DCT features shape:", dct_features.shape)

        # 梯度特征提取
        grad_features = self.grad_feature_extractor(x_grad)
        #print("Gradient features shape:", grad_features.shape)

        # 特征融合
        combined_features = torch.cat([dct_features, grad_features], dim=1)
        #print("Combined features shape:", combined_features.shape)

        # 动态路由
        capsule_output = self.capsule_layer(combined_features)
        #print("Capsule output shape before reshape:", capsule_output.shape)

        # 确保capsule_output被正确地展平
        capsule_output = capsule_output.view(x_dct.size(0), -1)  # 展平capsule_output
        #print("Capsule output shape after reshape:", capsule_output.shape)

        # 分类
        outputs = self.classifier(capsule_output)
        return outputs




def ResNet18():
    return FusionCapsNet()


#net = ResNet18().cuda()
#net = ResNet18().to(device)



# 在定义模型之后使用 DataParallel
net = ResNet18()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 使用 DataParallel 来包装模型
    net = nn.DataParallel(net)

# 将模型移至设备
net.to(device)


import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

# WarmUpLR类定义
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

# AverageMeter类定义
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

# accuracy函数定义
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
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (dct_images, grad_images, labels) in enumerate(trainloader):
        dct_images, grad_images, labels = dct_images.to(device), grad_images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(dct_images, grad_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if epoch < warm:
            scheduler.step()

    train_loss = running_loss / len(trainloader)
    acc = 100. * correct / total
    return train_loss, acc

def validate(val_loader, model, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (dct_images, grad_images, labels) in enumerate(val_loader):
            dct_images, grad_images, labels = dct_images.to(device), grad_images.to(device), labels.to(device)
            outputs = model(dct_images, grad_images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    return val_loss, acc




warm = 1
epoch = 160
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
# 设置超参数


# 集成学习的主要部分
def ensemble_predict(models, dataloader, device):
    # 确保所有模型都处于评估模式
    for model in models:
        model.eval()

    all_preds = []
    with torch.no_grad():
        for i, (dct_images, grad_images, labels) in enumerate(dataloader):
            dct_images, grad_images = dct_images.to(device), grad_images.to(device)
            preds = []
            for model in models:
                if isinstance(model, FusionCapsNet):  # 对于特定的FusionCapsNet模型
                    outputs = model(dct_images, grad_images)
                else:
                    # 对于EfficientNet和ResNeXt，我们只使用dct_images
                    outputs = model(dct_images)
                preds.append(outputs.cpu().numpy())
            # 汇总所有模型的预测
            all_preds.append(np.mean(preds, axis=0))

    # 计算最终预测结果
    final_preds = np.argmax(np.vstack(all_preds), axis=1)
    return final_preds







def main():
    print("Starting the training and validation process...")

    # 数据集的预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整为适合EfficientNet和ResNeXt的尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的标准化参数
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整为适合EfficientNet和ResNeXt的尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的标准化参数
    ])

    # 数据加载器
    train_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/progan_train', transform=train_transform)
    val_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/CNN_synth_testset', transform=val_transform)

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型
    fusion_caps_net = FusionCapsNet().to(device)
    efficient_net = EfficientNetModel().to(device)
    resnext_net = ResNeXtModel().to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    models = [fusion_caps_net, efficient_net, resnext_net]
    model_names = ['FusionCapsNet', 'EfficientNet', 'ResNeXt']

    for model, name in zip(models, model_names):
        print(f"\nTraining and validating model: {name}")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        for epoch in range(200):  # 假定训练200个epoch
            print(f"\nEpoch {epoch+1}/{200}")

            train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, scheduler, warm)
            print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

            val_loss, val_acc = validate(valloader, model, criterion)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

            scheduler.step()

    # 集成学习的预测和评估
    ensemble_acc = ensemble_predict(models, valloader, device)
    print(f"\nEnsemble Model Accuracy: {ensemble_acc:.2f}%")

if __name__ == "__main__":
    main()
