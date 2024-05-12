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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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


from torchvision.transforms import Grayscale
# ProGAN_Dataset类定义
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
                # 如果是目录，则递归调用
                self._load_images(path)
            elif path.endswith('.png'):
                # 判断是真实图片还是假图片
                label = 1 if '1_fake' in path else 0
                self.image_paths.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        dct_image = apply_dct(image)  # 应用DCT变换
        gradient_image = generate_gradient_image(image)  # 生成梯度图像
        gradient_image = Grayscale(num_output_channels=3)(gradient_image)  # 将梯度图像转换为三通道图像

        if self.transform:
            dct_image = self.transform(dct_image)
            gradient_image = self.transform(gradient_image)

        label = self.labels[idx]
        return dct_image, gradient_image, label


class DualBranchNetwork(nn.Module):
    def __init__(self):
        super(DualBranchNetwork, self).__init__()
        # DCT Branch
        self.dct_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Gradient Branch
        self.gradient_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Combined Layers
        self.combined_fc = nn.Sequential(
            nn.Linear(32 * 2 * 32 * 32, 512),  # Adjust the size according to your pooling and input image size
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, dct_input, gradient_input):
        dct_features = self.dct_branch(dct_input)
        gradient_features = self.gradient_branch(gradient_input)
        # Flatten and concatenate features
        dct_features = dct_features.view(dct_features.size(0), -1)
        gradient_features = gradient_features.view(gradient_features.size(0), -1)
        combined_features = torch.cat((dct_features, gradient_features), dim=1)
        output = self.combined_fc(combined_features)
        return output


# 数据集的预处理操作
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 适用于三通道图像
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
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention_layer = AttentionLayer(64)  # 添加注意力层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_attention=True)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use_attention=True)

        # 更新特征图尺寸
        # 假设输入尺寸为64x64，且使用了2x2池化层
        self.feature_size = 512 * block.expansion * 4 * 4  # 这里的4*4是假设的输出特征图尺寸

        self.linear = nn.Linear(self.feature_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, use_attention=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_attention=use_attention))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.attention_layer(out)  # 应用注意力层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 动态计算特征图尺寸
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        # 确保线性层输入尺寸正确
        if self.linear.in_features != out.shape[1]:
            self.linear = nn.Linear(out.shape[1], self.linear.out_features).cuda()

        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

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

# train函数定义
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (dct_images, gradient_images, labels) in enumerate(train_loader):
        dct_images, gradient_images, labels = dct_images.to(device), gradient_images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(dct_images, gradient_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    acc = 100. * correct / total
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc: {acc:.2f}%')

def validate(val_loader, model, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (dct_images, gradient_images, labels) in enumerate(val_loader):
            dct_images, gradient_images, labels = dct_images.to(device), gradient_images.to(device), labels.to(device)
            outputs = model(dct_images, gradient_images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    print(f'Validation Loss: {val_loss:.3f}, Validation Acc: {acc:.2f}%')
    return val_loss, acc




warm = 1
epoch = 160
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
# 设置超参数
# ...之前的代码保持不变...

def main():
    # 实例化训练集
    train_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/progan_train', transform=transform)
    
    # 实例化验证集
    val_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/CNN_synth_testset', transform=transform)

    # 创建数据加载器
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=5, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=5, pin_memory=True)

    # 实例化模型并将其移动到指定的GPU
    net = ResNet18().to(device)

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # 学习率调度器
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
    
    # 训练和验证
    best_acc = 0
    for epoch in range(160):
        # 训练
        train_loss, train_acc = train(trainloader, net, loss_function, optimizer, epoch, warmup_scheduler, warm)
        
        # 验证
        val_loss, val_acc = validate(valloader, net, loss_function)

        # 更新学习率
        if epoch >= warm:
            train_scheduler.step()

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            # 可以在这里添加保存模型的代码

    print(f'Best Validation Accuracy: {best_acc:.3f}%')


num_epochs=160
def main():
    transform = Compose([
        Resize((128, 128)),  # Adjust the size as needed
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust as needed
    ])

    train_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/progan_train', transform=transform)
    val_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/CNN_synth_testset', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = DualBranchNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        validate(val_loader, model, criterion)

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()