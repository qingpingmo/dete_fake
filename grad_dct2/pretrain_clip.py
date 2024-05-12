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
from sklearn.metrics import average_precision_score
import clip
from sklearn.metrics import average_precision_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_average_precision_score(val_loader, model):
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for i, (dct_images, grad_images, labels, orig_images) in enumerate(val_loader):
            dct_images, grad_images, orig_images, labels = dct_images.to(device), grad_images.to(device), orig_images.to(device), labels.to(device)
            outputs = model(dct_images, grad_images, orig_images)
            scores = F.softmax(outputs, dim=1)[:, 1]  # Get the scores for the positive class
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    average_precision = average_precision_score(all_labels, all_scores)
    return average_precision

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
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2(avg_out)
        max_out = self.fc1(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2(max_out)
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
        #print(f"CBAM forward x shape: {x.shape}")  # 添加的打印语句
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
        orig_image = Image.open(img_path).convert('RGB')
        if self.transform:
            dct_image = self.transform(dct_image)
            grad_image = self.transform(grad_image)  
            orig_image = self.transform(orig_image)

        label = self.labels[idx]
        return dct_image, grad_image, label, orig_image

transform = Compose([
    Resize((224, 224)),  # 修改为CLIP期望的输入尺寸
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
        # 加载预训练的CLIP模型
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        # 冻结CLIP模型的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
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

        self.cbam = CBAM(in_planes=256) 
        self.capsule_layer = CapsuleLayer(
            num_capsules=10,
            num_route_nodes=-1,
            in_channels=512, 
            out_channels=32,
            kernel_size=None,
            stride=None
        )

        self.classifier = nn.Linear(401920, 2) 

    def forward(self, x_dct, x_grad, x_orig):
        dct_features = self.dct_feature_extractor(x_dct)
        grad_features = self.grad_feature_extractor(x_grad)

        # 对 dct_features 和 grad_features 使用 CBAM
        dct_features = self.cbam(dct_features)
        grad_features = self.cbam(grad_features)

        # 将 dct_features 和 grad_features 展平为 2 维
        dct_features = dct_features.view(dct_features.size(0), -1)
        grad_features = grad_features.view(grad_features.size(0), -1)

        with torch.no_grad():
            clip_features = self.clip_model.encode_image(x_orig)

        # 将处理后的 dct_features 和 grad_features 与 clip_features 拼接
        combined_features = torch.cat([dct_features, grad_features, clip_features], dim=1)

        # 打印特征形状
        #print(f"combined_features shape: {combined_features.shape}")

        # 使用分类器
        outputs = self.classifier(combined_features)
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
    # 确保在循环开始时模型处于训练模式
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (dct_images, grad_images, labels, orig_images) in enumerate(trainloader):
        # 将数据移动到正确的设备上
        dct_images, grad_images, orig_images, labels = dct_images.to(device), grad_images.to(device), orig_images.to(device), labels.to(device)

        # 优化器梯度归零
        optimizer.zero_grad()

        # 通过模型前向传播
        outputs = model(dct_images, grad_images, orig_images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 如果处于预热阶段，则更新预热调度器
        if epoch < warm:
            scheduler.step()

    # 计算平均损失和准确率
    train_loss = running_loss / len(trainloader)
    acc = 100. * correct / total

    return train_loss, acc

def validate(val_loader, model, criterion):
    # 确保在验证开始时模型处于评估模式
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (dct_images, grad_images, labels, orig_images) in enumerate(val_loader):
            # 将数据移动到正确的设备上
            dct_images, grad_images, orig_images, labels = dct_images.to(device), grad_images.to(device), orig_images.to(device), labels.to(device)

            # 通过模型前向传播
            outputs = model(dct_images, grad_images, orig_images)

            # 计算损失
            loss = criterion(outputs, labels)

            # 统计
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # 计算平均损失和准确率
    val_loss = val_loss / len(val_loader)
    acc = 100. * correct / total

    return val_loss, acc


def load_pretrained_model(model, path):
    # 加载预训练的状态字典
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    
    # 过滤出匹配的权重
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    
    # 更新当前模型的状态字典
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)



warm = 1
epoch = 160
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

def main():
    train_dataset = ProGAN_Dataset(root_dir='/opt/data/private/wangjuntong/datasets/progan_train', transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=5, pin_memory=True)
    net = ResNet18().to(device)
    pretrained_model_path = 'wjtmodel.pth'  # 这里是您预训练模型的路径
    load_pretrained_model(net, pretrained_model_path)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
    best_acc = 0
    for epoch in range(200):
        train_loss, train_acc = train(trainloader, net, loss_function, optimizer, epoch, warmup_scheduler, warm)
        if epoch >= warm:
            train_scheduler.step()

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}')

        folders = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
        for folder in folders:
            print(f"Validating on {folder}...")
            val_dataset_folder = os.path.join('/opt/data/private/wangjuntong/datasets/CNN_synth_testset', folder)
            val_dataset = ProGAN_Dataset(root_dir=val_dataset_folder, transform=transform)
            valloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=5, pin_memory=True)

            val_loss, val_acc = validate(valloader, net, loss_function)
            val_ap = get_average_precision_score(valloader, net)  # Calculate the average precision score
            print(f'Folder: {folder}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val AP: {val_ap:.3f}')

            if val_acc > best_acc:
                best_acc = val_acc

        print(f'Best Validation Accuracy: {best_acc:.3f}%')

if __name__ == "__main__":
    main()