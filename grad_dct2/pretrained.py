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
from PIL import ImageFilter, ImageOps



os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
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
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 数据集的根目录
        self.transform = transform  # 预处理转换
        self.image_paths = []  # 存储图像路径的列表
        self.labels = []  # 存储图像标签的列表
        self._load_images(self.root_dir)  # 加载图像

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
        img_path = self.image_paths[idx]  # 根据索引获取图像路径
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB格式
        dct_image = apply_dct(image)  # 对图像应用DCT变换
        grad_image = generate_gradient_image(image)  # 生成图像的梯度图

        if self.transform:  # 如果设置了预处理转换
            dct_image = self.transform(dct_image)  # 对DCT图像应用预处理转换
            grad_image = self.transform(grad_image)  # 对梯度图应用预处理转换

        label = self.labels[idx]  # 获取图像标签
        return dct_image, grad_image, label  # 返回DCT图像、梯度图和标签
    


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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_attention=False, heads=4, attention_type='se'):
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
        self.attention_type = attention_type
        if self.use_attention:
            if self.attention_type == 'se':
                self.attention = SELayer(planes)
            elif self.attention_type == 'mhsa':
                self.attention = MultiHeadSelfAttention(planes, heads)
            else:
                raise NotImplementedError("Attention type not supported")

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if self.use_attention:
            out = self.attention(out)

        out = F.relu(out)
        return out

class FusionCapsNet(nn.Module):
    def __init__(self):
        super(FusionCapsNet, self).__init__()
        self.dct_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.grad_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
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

    def forward(self, x_dct, x_grad):
        dct_features = self.dct_feature_extractor(x_dct)
        grad_features = self.grad_feature_extractor(x_grad)
        combined_features = torch.cat([dct_features, grad_features], dim=1)
        combined_features = self.cbam(combined_features)
        capsule_output = self.capsule_layer(combined_features)
        capsule_output = capsule_output.view(x_dct.size(0), -1)
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

    for i, (dct_images, grad_images, labels) in enumerate(trainloader):  # 遍历训练数据
        dct_images, grad_images, labels = dct_images.to(device), grad_images.to(device), labels.to(device)  # 将数据移动到指定的设备
        optimizer.zero_grad()  # 清空梯度

        outputs = model(dct_images, grad_images)  # 计算模型输出
        loss = criterion(outputs, labels)  # 计算损失
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
    model.eval()  # 将模型设置为评估模式
    val_loss = 0  # 验证损失
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数
    all_predicted = []  # 存储所有预测值
    all_targets = []  # 存储所有真实标签

    with torch.no_grad():  # 不计算梯度
        for i, (dct_images, grad_images, labels) in enumerate(val_loader):  # 遍历验证数据
            dct_images, grad_images, labels = dct_images.to(device), grad_images.to(device), labels.to(device)  # 将数据移动到指定的设备
            outputs = model(dct_images, grad_images)  # 计算模型输出
            loss = criterion(outputs, labels)  # 计算损失

            val_loss += loss.item()  # 累加损失
            _, predicted = outputs.max(1)  # 获取预测的类别
            total += labels.size(0)  # 更新总样本数
            correct += predicted.eq(labels).sum().item()  # 更新正确预测的数量
            all_predicted.extend(predicted.cpu().numpy())  # 添加预测值到列表
            all_targets.extend(labels.cpu().numpy())  # 添加真实标签到列表

    val_loss = val_loss / len(val_loader)  # 计算平均验证损失
    acc = 100. * correct / total  # 计算验证准确率
    ap = average_precision_score(all_targets, all_predicted)  # 计算平均精度分数
    return val_loss, acc, ap  # 返回验证损失、准确率和平均精度分数





def save_model(model, path):
    """保存模型的状态字典"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """加载模型的状态字典"""
    model.load_state_dict(torch.load(path))



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
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=5, pin_memory=True)
    # 实例化模型并移动到指定的设备
    net = ResNet18().to(device)
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # 定义学习率调度器
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    # 创建学习率预热调度器
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
    # 初始化最佳准确率
    best_acc = 0.0
    best_avg_acc = 0.0  # Initialize best_avg_acc to 0
    best_epoch = -1
    save_path = 'wjtmodel.pth'
    # 开始训练周期
    for epoch in range(200):
        # 训练模型
        train_loss, train_acc = train(trainloader, net, loss_function, optimizer, epoch, warmup_scheduler, warm)
        # 如果结束预热，更新学习率
        if epoch >= warm:
            train_scheduler.step()

        # 打印训练结果
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}')

    # 定义待验证的数据集文件夹列表
        all_val_accs = []
        folders = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
        # 遍历每个文件夹进行验证
        for folder in folders:
            print(f"Validating on {folder}...")
            # 构建验证数据集路径
            val_dataset_folder = os.path.join('/opt/data/private/wangjuntong/datasets/CNN_synth_testset', folder)
            # 创建验证数据集实例
            val_dataset = ProGAN_Dataset(root_dir=val_dataset_folder, transform=transform)
            # 创建验证数据加载器
            valloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=5, pin_memory=True)

            # 验证模型
            val_loss, val_acc, val_ap = validate(valloader, net, loss_function)
            # 打印验证结果
            print(f'Folder: {folder}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val AP: {val_ap:.3f}')
            all_val_accs.append(val_acc)

            # 更新最佳准确率
        avg_val_acc = sum(all_val_accs) / len(all_val_accs)
    
        if avg_val_acc > best_avg_acc:
            print(f'Saving new best model at epoch {epoch} with Avg Val Acc: {avg_val_acc:.3f}')
            # Save the model
            save_model(net, save_path)
            # Update the best average validation accuracy and corresponding epoch
            best_avg_acc = avg_val_acc
            best_epoch = epoch
        # 打印最佳验证准确率
    print(f'Best Average Validation Accuracy: {best_avg_acc:.3f}% at Epoch {best_epoch}')

# 程序入口
if __name__ == "__main__":
    main()  # 执行main函数