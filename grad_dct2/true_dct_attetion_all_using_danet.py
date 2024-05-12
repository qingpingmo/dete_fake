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
import torch.nn.functional as F
import gc
from torch.nn import DataParallel

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)





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

    # 检查最大值和最小值是否相等，以避免除以零
    min_val = np.min(idct_transformed)
    max_val = np.max(idct_transformed)
    if max_val - min_val > 0:
        idct_transformed_normalized = (idct_transformed - min_val) / (max_val - min_val)
        image_uint8 = (idct_transformed_normalized * 255).astype(np.uint8)
        return Image.fromarray(image_uint8)
    else:
        # 如果最大值和最小值相等，则直接返回原始图像
        return image


# ProGAN_Dataset类定义
class ProGAN_Dataset(Dataset):
    def __init__(self, root_dir, categories=['car', 'cat', 'chair', 'horse'], transform=None):
        self.transform = transform
        self.image_tensors = []
        self.labels = []
        self.categories = categories
        self._load_images(root_dir)

    def _load_images(self, current_dir):
        for item in os.listdir(current_dir):
            path = os.path.join(current_dir, item)
            if os.path.isdir(path) and any(cat in path for cat in self.categories):
                # 如果是目标类别的目录，则递归调用
                self._load_images(path)
            elif path.endswith('.png'):
                # 判断是真实图片还是假图片
                label = 1 if '1_fake' in path else 0
                try:
                    image = Image.open(path).convert('RGB')
                    image = apply_dct(image)
                    if self.transform:
                        image = self.transform(image)
                    self.image_tensors.append(image)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.labels[idx]

    def __len__(self):
        return len(self.image_tensors)





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

    def __init__(self, in_planes, planes, stride=1):
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
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
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 更新特征图尺寸
        # 假设输入尺寸为64x64，且使用了2x2池化层
        self.feature_size = 512 * block.expansion * 4 * 4  # 这里的4*4是假设的输出特征图尺寸

        self.linear = nn.Linear(self.feature_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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

net = ResNet18().cuda()

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


class DaNetResNet(nn.Module):
    def __init__(self, in_channels, danet_channels, resnet):
        super(DaNetResNet, self).__init__()
        # 新增一个卷积层来改变输入通道数
        self.input_conv = nn.Conv2d(in_channels, danet_channels, kernel_size=1)
        self.danet_head = _DAHead(danet_channels, nclass=2)
        self.resnet = resnet

        # 添加一个转换层以匹配通道数
        self.transition_conv = nn.Conv2d(2, 3, kernel_size=1)  # 假设danet_output有2个通道

    def forward(self, x):
        x = self.input_conv(x)  # 转换输入通道数
        danet_output = self.danet_head(x)[0]
        danet_output = self.transition_conv(danet_output)  # 转换通道数以匹配ResNet
        resnet_output = self.resnet(danet_output)
        return resnet_output





# train函数定义
def train(trainloader, model, criterion, optimizer, epoch, scheduler, warm):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 释放不再需要的变量
        del inputs, labels, outputs, loss
        torch.cuda.empty_cache()  # 清除未被使用的内存

        if epoch < warm:
            scheduler.step()

    train_loss = running_loss / len(trainloader)
    acc = 100. * correct / total

    # 手动启动垃圾回收器
    gc.collect()

    return train_loss, acc

# 修改validate函数
def validate(val_loader, model, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

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
def main():
    # 实例化训练集和验证集
    train_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/progan_train', transform=transform)
    val_dataset = ProGAN_Dataset(root_dir='/root/rstao/datasets/CNN_synth_testset', transform=transform)

    # 创建数据加载器
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=5, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=5, pin_memory=True)

    # 设置可用的GPU
    torch.cuda.set_device('cuda:0')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # 实例化模型并应用DataParallel
    resnet = ResNet18()
    model = DaNetResNet(in_channels=3, danet_channels=512, resnet=resnet)
    if torch.cuda.is_available():
        model.cuda()
        model = DataParallel(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    # 训练和验证循环
    best_acc = 0
    for epoch in range(160):
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, warmup_scheduler, warm)
        val_loss, val_acc = validate(valloader, model, criterion)

        if epoch >= warm:
            train_scheduler.step()

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            # 在这里添加保存模型的代码

    print(f'Best Validation Accuracy: {best_acc:.3f}%')

if __name__ == "__main__":
    main()