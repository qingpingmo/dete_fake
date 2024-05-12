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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.multiprocessing as mp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def generate_gradient_image(image_tensor):
    """
    生成图像的梯度图像。
    """
    # 确保在转换前将图像张量移至CPU，并转换为NumPy数组
    image_np = image_tensor.cpu().numpy()
    # 需要调整张量的形状以匹配NumPy数组的预期维度（C, H, W）-> (H, W, C)
    image_np = np.transpose(image_np, (1, 2, 0))

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad = np.zeros_like(image_np[:,:,0])  # 只取一个通道
    for i in range(image_np.shape[2]):  # 对每个颜色通道分别应用
        grad_x = np.abs(convolve2d(image_np[:, :, i], sobel_x, mode='same', boundary='wrap'))
        grad_y = np.abs(convolve2d(image_np[:, :, i], sobel_y, mode='same', boundary='wrap'))
        grad += np.sqrt(grad_x ** 2 + grad_y ** 2)  # 累加不同通道的梯度

    # 标准化到0-255并转换为uint8
    grad = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    gradient_uint8 = (grad * 255).astype(np.uint8)

    # 将处理后的NumPy数组转换回张量，并发送回原始设备
    gradient_tensor = torch.tensor(gradient_uint8, dtype=torch.float).unsqueeze(0).to(device)  # 添加一个通道维度

    return gradient_tensor




class ResNetBranch(nn.Module):
    # 定义一个ResNet分支
    def __init__(self, block, num_blocks, in_channels, use_attention=False):
        super(ResNetBranch, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention_layer = AttentionLayer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_attention=use_attention)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_attention=use_attention)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_attention=use_attention)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use_attention=use_attention)

    def _make_layer(self, block, planes, num_blocks, stride, use_attention=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_attention=use_attention))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.attention_layer(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class DualBranchResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(DualBranchResNet, self).__init__()
        self.dct_branch = ResNetBranch(block, num_blocks, in_channels=3, use_attention=True)
        self.gradient_branch = ResNetBranch(block, num_blocks, in_channels=1, use_attention=True)
        self.fc = nn.Linear(512 * block.expansion * 2, 2)  # 假设两个分支的特征图大小相同

    def forward(self, dct_input, gradient_input):
        dct_out = self.dct_branch(dct_input)
        gradient_out = self.gradient_branch(gradient_input)

        # 合并两个分支的特征
        out = torch.cat((dct_out, gradient_out), dim=1)

        # 全局平均池化
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        # 全连接层
        out = self.fc(out)
        return out


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

def apply_dct(image_tensor, keep_ratio=0.1):
    # 确保在转换前将图像张量移至CPU
    # 确保图像张量在CPU上
    image_np = image_tensor.cpu().numpy()
    # 需要调整张量的形状以匹配numpy数组的预期维度（C, H, W）-> (H, W, C)
    image_np = np.transpose(image_np, (1, 2, 0))
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
    image_tensor_dct = torch.tensor(np.transpose(image_np, (2, 0, 1)), dtype=torch.float).to(device)
    return image_tensor_dct


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

        if self.transform:
            image = self.transform(image)  # 确保图像首先被转换为张量

        # 应用DCT变换（现在在张量上操作）
        image_dct = apply_dct(image)

        label = self.labels[idx]
        return image_dct, label


# 数据集的预处理操作
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
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
    return DualBranchResNet(BasicBlock, [2, 2, 2, 2])

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

accumulation_steps = 4  # 梯度累积步数

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()  # 初始化梯度

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        dct_inputs = torch.stack([apply_dct(input) for input in inputs]).to(device)
        gradient_inputs = torch.stack([generate_gradient_image(input) for input in inputs]).to(device)

        outputs = model(dct_inputs, gradient_inputs)
        loss = criterion(outputs, targets) / accumulation_steps  # 标准化损失
        loss.backward()  # 累积梯度

        if (batch_idx + 1) % accumulation_steps == 0:  # 每accumulation_steps步更新一次
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # 反标准化累积的损失
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Train Epoch: {epoch} | Loss: {train_loss:.3f} | Acc: {train_acc:.3f}% ({correct}/{total})')



# validate函数定义
def validate(val_loader, model, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 分别处理输入数据以适应两个分支
            dct_inputs = torch.stack([apply_dct(input) for input in inputs]).to(device)
            gradient_inputs = torch.stack([generate_gradient_image(input) for input in inputs]).to(device)

            outputs = model(dct_inputs, gradient_inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = total_loss / len(val_loader)
    val_acc = 100. * correct / total
    print(f'Validation | Loss: {val_loss:.3f} | Acc: {val_acc:.3f}% ({correct}/{total})')
    return val_acc



warm = 1
epoch = 160
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
# 设置超参数
def main():
    global best_acc
    best_acc = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # 数据预处理设置
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载训练和验证数据集
    train_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/progan_train', transform=transform)
    val_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/CNN_synth_testset', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=5)

    # 初始化模型
    model = ResNet18().to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # 训练和验证循环
    for epoch in range(start_epoch, start_epoch + 200):
        train(train_loader, model, criterion, optimizer, epoch)
        val_acc = validate(val_loader, model, criterion)

        # 保存最佳模型
        if val_acc > best_acc:
            print('Saving best model...')
            state = {
                'model': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = val_acc

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
