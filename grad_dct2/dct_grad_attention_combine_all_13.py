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
from PIL import ImageOps
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ImageProcessor(nn.Module):
    def __init__(self, attention_heads=4):  # 将heads数量设置为1
        super(ImageProcessor, self).__init__()
        embed_size = 32  # 保持embed_size为3
        # 为DCT和梯度图像使用不同的输入维度
        self.attention_dct = MultiHeadSelfAttention(embed_size=embed_size, heads=attention_heads, input_dim=3)
        self.attention_grad = MultiHeadSelfAttention(embed_size=embed_size, heads=attention_heads, input_dim=1)


    

    def forward(self, image):
    # Convert PIL Image to tensor and apply transformations
        embed_size = 3  # 保持embed_size为3
        attention_heads=1
        if not isinstance(image, torch.Tensor):
            transform = ToTensor()
            image = transform(image)

        
        image_dct = apply_dct(image)
        image_grad = generate_gradient_image(image)

    # 此处假设 image_dct 和 image_grad 都是 [B, C, H, W] 形状的张量
        # 假设 image_dct 和 image_grad 的形状都是 [B, C, H, W]
        B, C, H, W = image_dct.size()

        # 平铺图像
        image_dct_flat = image_dct.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        image_grad_flat = image_grad.view(B, 1, H * W).transpose(1, 2)  # [B, H*W, 1]

        # 为注意力层传递正确的 input_dim
        self.attention_dct = MultiHeadSelfAttention(embed_size=embed_size, heads=attention_heads, input_dim=C)
        self.attention_grad = MultiHeadSelfAttention(embed_size=embed_size, heads=attention_heads, input_dim=1)

        # 应用注意力机制
        image_dct_att = self.attention_dct(image_dct_flat, image_dct_flat, image_dct_flat, None)
        image_grad_att = self.attention_grad(image_grad_flat, image_grad_flat, image_grad_flat, None)


    # Feature fusion
        # Detach and clone the output
        combined_feature = torch.cat((image_dct_att, image_grad_att), dim=1)
        return combined_feature.detach().clone()








def generate_gradient_image(image):
    # Convert PIL Image to NumPy array
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
    
    # Convert to tensor, ensure single channel
    tensor_image = ToTensor()(gradient_uint8)  # This will be [1, H, W]
    return tensor_image.unsqueeze(0)  # Adds a batch dimension, final shape [B, 1, H, W]



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, input_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 确保input_dim正确设置
        self.input_dim = input_dim

        # 使用input_dim作为线性层的输入尺寸
        self.values = nn.Linear(self.input_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.input_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.input_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = self.values(values).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).view(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out






class SpatialAttention(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = F.avg_pool2d(x, x.size(2))
        max_out, _ = F.max_pool2d(x, x.size(2))
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

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
    idct_transformed_normalized = (idct_transformed - np.min(idct_transformed)) / (np.max(idct_transformed) - np.min(idct_transformed) + 1e-8)
    image_uint8 = (idct_transformed_normalized * 255).astype(np.uint8)
    # Convert to tensor, ensure single channel
    tensor_image = ToTensor()(image_uint8)  # This will be [C, H, W] where C is 1
    return tensor_image.unsqueeze(0)  # Adds a batch dimension, final shape [B, C, H, W]


# ProGAN_Dataset类定义
class ProGAN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self._load_images(self.root_dir)
        self.image_processor = ImageProcessor()  # 创建 ImageProcessor 实例


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

        # 先应用 transform（如果是针对 PIL Image 的）
        if self.transform:
            image = self.transform(image)

        # 然后应用 ImageProcessor
        processed_image = self.image_processor(image).detach()

        label = self.labels[idx]
        return processed_image, label





# 数据集的预处理操作
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    # 由于现在是单通道图像，因此mean和std应该是单个值而不是三元组
    Normalize(mean=[0.5], std=[0.5])
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
            self.spatial_attention = SpatialAttention(planes)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))

        # Apply Multi-Head Self Attention
        if self.use_attention:
            out = out.view(out.shape[0], -1, out.shape[1])  # Adjust shape to match self-attention layer input
            out = self.attention(out, out, out, None)
            out = out.view(out.shape[0], out.shape[2], int(sqrt(out.shape[1])), int(sqrt(out.shape[1])))  # Restore shape

        # Apply Spatial Attention
        if self.use_attention:
            spatial_attention = self.spatial_attention(out)
            out = out * spatial_attention

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet类
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # 修改为接受6通道输入（两个RGB图像堆叠）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, use_attention=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_attention=use_attention))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 修改平均池化层以适应输出特征图的维度
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    num_features = 512 * 4 * 4 # 512是最后一个卷积层的输出通道数, 4x4是特征图尺寸
    return ResNet(BasicBlock, [2, 2, 2, 2], num_features)

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
# train函数定义
def train(trainloader, model, criterion, optimizer, epoch, scheduler, warm):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader):
        # 将数据移至GPU
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
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

# validate函数定义
def validate(val_loader, model, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

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
# ...之前的代码保持不变...

def main():
    # 实例化训练集
    train_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/progan_train', transform=transform)
    
    # 实例化验证集
    val_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/CNN_synth_testset', transform=transform)

    # 创建数据加载器
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

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

if __name__ == "__main__":
    main()