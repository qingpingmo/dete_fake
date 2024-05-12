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
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class BidirectionalFusionFPN(nn.Module):
    def __init__(self, input_channels, feature_sizes=[256, 256], lstm_hidden_size=256):
        super(BidirectionalFusionFPN, self).__init__()
        self.input_channels = input_channels
        self.feature_sizes = feature_sizes

        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(in_channels, fs, kernel_size=1) for in_channels, fs in zip(input_channels, feature_sizes)
        ])

        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(fs, fs, kernel_size=3, padding=1) for fs in feature_sizes
        ])

        self.bilstm = nn.LSTM(input_size=feature_sizes[-1], hidden_size=lstm_hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)

        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest') for _ in range(len(feature_sizes) - 1)
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))

        # 新增匹配通道数的卷积层
        self.match_channels_layer = nn.Conv2d(512, 256, kernel_size=1)  # 从512通道调整到256通道

    def forward(self, inputs):
        input_tensor = inputs[0]  # Assume inputs is a list with one tensor for concatenated features

        reduced_input = self.reduce_layers[0](input_tensor)
        #print(f"Reduced input shape: {reduced_input.shape}")

        seq_features = reduced_input.permute(0, 2, 3, 1).contiguous().view(reduced_input.size(0), -1, reduced_input.size(1))
        #print(f"Seq features shape: {seq_features.shape}")

        bilstm_output, _ = self.bilstm(seq_features)
        #print(f"BILSTM output shape: {bilstm_output.shape}")

        features = bilstm_output.reshape(reduced_input.size(0), reduced_input.size(2), reduced_input.size(3), -1).permute(0, 3, 1, 2)
        #print(f"Restored features shape: {features.shape}")

        # 在平滑卷积之前应用匹配通道数的卷积层
        features = self.match_channels_layer(features)

        for i in range(len(self.feature_sizes) - 1):
            upsampled = self.upsample_layers[i](features)
            #print(f"Upsampled features shape (iteration {i}): {upsampled.shape}")
            if upsampled.size()[2:] != features.size()[2:]:
                upsampled = self.adaptive_pool(upsampled)
                #print(f"Adaptively pooled features shape (iteration {i}): {upsampled.shape}")
            features = features + upsampled

        features = self.smooth_layers[0](features)
        #print(f"Smoothed features shape: {features.shape}")

        return features












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

class FusionCapsNetWithFPN(nn.Module):
    def __init__(self):
        super(FusionCapsNetWithFPN, self).__init__()
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

        # FPN module
        # In FusionCapsNetWithFPN.__init__
        self.fpn = BidirectionalFusionFPN(input_channels=[512], feature_sizes=[256, 256], lstm_hidden_size=256)  # Adjusted for single input


        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Linear(256, 2)

    def forward(self, x_dct, x_grad):
        dct_features = self.dct_feature_extractor(x_dct)
        #print(f"dct_features shape: {dct_features.shape}")  # 打印DCT特征维度

        grad_features = self.grad_feature_extractor(x_grad)
        #print(f"grad_features shape: {grad_features.shape}")  # 打印梯度特征维度

        combined_features = torch.cat([dct_features, grad_features], dim=1)
        #print(f"combined_features shape: {combined_features.shape}")  # 打印合并特征维度

        combined_features = self.cbam(combined_features)
        #print(f"combined_features after CBAM shape: {combined_features.shape}")  # 打印经过CBAM处理后的特征维度

        # Pass concatenated features to FPN
        features = self.fpn([combined_features])  # Adjusted for single input
        #print(f"features from FPN shape: {features.shape}")  # 打印FPN输出特征维度

        pooled_features = self.global_avg_pool(features)
        #print(f"pooled_features shape: {pooled_features.shape}")  # 应该是 [batch_size, 256, 1, 1]

        # 不需要循环，直接扁平化
        flat_features = pooled_features.view(pooled_features.size(0), -1)
        #print(f"flat_features shape: {flat_features.shape}")  # 应该是 [batch_size, 256]

        outputs = self.classifier(flat_features)
        #print(f"outputs shape: {outputs.shape}")
        return outputs



def ResNet18():
    return FusionCapsNetWithFPN()


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
    all_predicted = []
    all_targets = []

    with torch.no_grad():
        for i, (dct_images, grad_images, labels) in enumerate(val_loader):
            dct_images, grad_images, labels = dct_images.to(device), grad_images.to(device), labels.to(device)
            outputs = model(dct_images, grad_images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    ap = average_precision_score(all_targets, all_predicted)
    return val_loss, acc, ap




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
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
    best_acc = 0
    for epoch in range(50):
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

        val_loss, val_acc, val_ap = validate(valloader, net, loss_function)
        print(f'Folder: {folder}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val AP: {val_ap:.3f}')

        if val_acc > best_acc:
            best_acc = val_acc
    print(f'Best Validation Accuracy: {best_acc:.3f}%')

if __name__ == "__main__":
    main()