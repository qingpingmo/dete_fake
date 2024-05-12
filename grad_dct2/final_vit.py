# 引入必要的包
import torch  # PyTorch库，用于深度学习
import torchvision.transforms as transforms  # torchvision库中的transforms模块，用于图像预处理
import os  # 用于操作系统功能，如文件路径操作
from PIL import Image  # PIL库（现在称为Pillow），用于图像处理
from torch.utils.data import Dataset, DataLoader  # 用于创建数据集和数据加载器
from torchvision.transforms import Compose, Resize, Normalize, ToTensor  # 引入更多图像预处理函数
import numpy as np  # NumPy库，用于数值计算
from scipy.fftpack import dct, idct  # SciPy库中的离散余弦变换和逆变换函数
import torch.nn as nn  # PyTorch中的神经网络模块
from math import sqrt  # math库中的平方根函数
from scipy.signal import convolve2d  # SciPy中的二维卷积函数
from sklearn.metrics import average_precision_score  # 从scikit-learn中引入平均精度分数计算函数


from transformers import ViTFeatureExtractor, ViTModel




# 设置CUDA环境变量，指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 根据CUDA可用性设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    def forward(self, images):
        # 使用特征提取器处理图像
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)  # 确保像素值在正确的设备上
        # 使用模型提取特征
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state




# 定义ChannelAttention类，实现通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),  # 1x1卷积用于降维
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 1x1卷积用于升维
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 对平均池化的结果应用1x1卷积
        max_out = self.fc(self.max_pool(x))  # 对最大池化的结果应用1x1卷积
        out = avg_out + max_out  # 将两个结果相加
        return self.sigmoid(out)  # 应用Sigmoid激活函数并返回结果

# 定义SpatialAttention类，实现空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)  # 用于计算空间注意力的卷积层
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算通道平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算通道最大值
        x = torch.cat([avg_out, max_out], dim=1)  # 将平均值和最大值沿通道维度拼接
        x = self.conv1(x)  # 应用卷积层
        return self.sigmoid(x)  # 应用Sigmoid激活函数并返回结果

# 定义CBAM类，结合通道注意力和空间注意力
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)  # 通道注意力模块
        self.spatial_attention = SpatialAttention(kernel_size)  # 空间注意力模块

    def forward(self, x):
        x = x * self.channel_attention(x)  # 应用通道注意力
        x = x * self.spatial_attention(x)  # 应用空间注意力
        return x  # 返回结果

# 定义CapsuleLayer类，实现胶囊网络层
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes  # 路由节点数量
        self.num_iterations = num_iterations  # 动态路由迭代次数
        kernel_size = kernel_size if kernel_size is not None else 3  # 卷积核大小
        stride = stride if stride is not None else 1  # 卷积步长
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
            for _ in range(num_capsules)  # 根据胶囊数量创建卷积层列表
        ])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)  # 计算张量的平方和
        scale = squared_norm / (1 + squared_norm)  # 缩放因子
        return scale * tensor / torch.sqrt(squared_norm)  # 应用squash激活函数

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]  # 计算所有胶囊的输出
            priors = torch.cat(priors, dim=-1)  # 在最后一个维度上拼接
            logits = torch.zeros(*priors.size()).to(x.device)  # 初始化路由权重

            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)  # 计算路由概率
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))  # 计算胶囊输出

                if i < self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)  # 更新路由权重
                    logits = logits + delta_logits

        else:
            outputs = [capsule(x) for capsule in self.capsules]  # 如果没有路由节点，直接计算所有胶囊的输出
            outputs = torch.stack(outputs, dim=1)  # 在第一个维度上堆叠
            outputs = outputs.view(x.size(0), -1, outputs.size(-1))  # 重塑输出
            outputs = self.squash(outputs)  # 应用squash激活函数

        return outputs  # 返回胶囊层的输出


# 定义apply_dct函数，用于对图像进行离散余弦变换
def apply_dct(image):
    image_np = np.array(image)  # 将图像转换为NumPy数组
    dct_transformed = dct(dct(image_np, axis=0, norm='ortho'), axis=1, norm='ortho')  # 对图像应用二维DCT
    dct_image = np.log(np.abs(dct_transformed) + 1)  # 使用对数尺度以便更好地可视化DCT变换结果
    dct_image_normalized = (dct_image - np.min(dct_image)) / (np.max(dct_image) - np.min(dct_image))  # 归一化
    image_uint8 = (dct_image_normalized * 255).astype(np.uint8)  # 转换为8位无符号整数
    return Image.fromarray(image_uint8)  # 将NumPy数组转换回图像

# 定义generate_gradient_image函数，用于生成图像的梯度图
def generate_gradient_image(image):
    image_np = np.array(image)  # 将图像转换为NumPy数组
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel算子（x方向）
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel算子（y方向）
    grad = np.zeros_like(image_np)  # 初始化梯度图像
    for i in range(3):  # 对每个颜色通道分别应用Sobel算子
        grad_x = np.abs(convolve2d(image_np[:, :, i], sobel_x, mode='same', boundary='wrap'))  # 计算x方向梯度
        grad_y = np.abs(convolve2d(image_np[:, :, i], sobel_y, mode='same', boundary='wrap'))  # 计算y方向梯度
        grad[:, :, i] = np.sqrt(grad_x ** 2 + grad_y ** 2)  # 计算总梯度

    grad = np.max(grad, axis=2)  # 在所有颜色通道中取最大值作为总梯度
    gradient_normalized = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))  # 归一化
    gradient_uint8 = (gradient_normalized * 255).astype(np.uint8)  # 转换为8位无符号整数

    return Image.fromarray(gradient_uint8)  # 将NumPy数组转换回图像

# 定义ProGAN_Dataset类，用于加载和处理ProGAN生成的数据集
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

# 定义图像预处理转换
transform = Compose([
    Resize((256, 256)),  # 将图像大小调整为256x256
    ToTensor(),  # 将图像转换为PyTorch张量
    Normalize(mean=[0.5], std=[0.5])  # 对张量进行归一化
])

# 创建ProGAN数据集的实例
progan_dataset = ProGAN_Dataset(root_dir='/opt/data/private/wangjuntong/datasets/progan_train', transform=transform)
# 计算训练集和验证集的大小
train_size = int(0.8 * len(progan_dataset))
val_size = len(progan_dataset) - train_size
# 将数据集分割为训练集和验证集
train_dataset, val_dataset = torch.utils.data.random_split(progan_dataset, [train_size, val_size])
# 定义批次大小
batch_size = 16
# 创建数据加载器
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

# 定义FusionCapsNet类，融合DCT特征、梯度特征和胶囊网络
class FusionCapsNet(nn.Module):
    def __init__(self):
        super(FusionCapsNet, self).__init__()
        self.dct_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # DCT特征提取器的第一个卷积层
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 最大池化层
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 第二个卷积层
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 第三个卷积层
            nn.ReLU(),  # ReLU激活函数
        )

        self.grad_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 梯度特征提取器的第一个卷积层
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 最大池化层
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 第二个卷积层
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 第三个卷积层
            nn.ReLU(),  # ReLU激活函数
        )

        self.vit_feature_extractor = ViTFeatureExtractor()  # ViT特征提取器

        self.cbam = CBAM(in_planes=512)  # CBAM注意力模块，用于融合DCT和梯度特征
        self.capsule_layer = CapsuleLayer(
            num_capsules=10,  # 胶囊数量
            num_route_nodes=-1,  # 没有路由节点
            in_channels=512,  # 输入通道数
            out_channels=32,  # 输出通道数
            kernel_size=None,  # 不使用卷积核
            stride=None  # 不使用步长
        )

        self.classifier = nn.Linear(288000, 2)  # 分类器，将胶囊网络的输出映射到2个类别

    def forward(self, x_dct, x_grad, x_vit):
        dct_features = self.dct_feature_extractor(x_dct)
        grad_features = self.grad_feature_extractor(x_grad)
        vit_features = self.vit_feature_extractor(x_vit)  # 提取ViT特征

        # 在通道维度上拼接DCT、梯度和ViT特征
        combined_features = torch.cat([dct_features, grad_features, vit_features.flatten(1)], dim=1)
        combined_features = self.cbam(combined_features)
        capsule_output = self.capsule_layer(combined_features)
        capsule_output = capsule_output.view(x_dct.size(0), -1)
        outputs = self.classifier(capsule_output)
        return outputs

# 创建ResNet18模型的实例
def ResNet18():
    return FusionCapsNet()  # 返回FusionCapsNet实例

# 实例化模型
net = ResNet18()
if torch.cuda.device_count() > 1:  # 如果有多个GPU
    print("Let's use", torch.cuda.device_count(), "GPUs!")  # 打印使用的GPU数量
    net = nn.DataParallel(net)  # 使用多GPU训练

net.to(device)  # 将模型移动到指定的设备
import torch.optim as optim  # 引入优化器模块
from torch.optim.lr_scheduler import _LRScheduler  # 引入学习率调度器的基类

# 定义WarmUpLR类，实现学习率预热
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters  # 总迭代次数
        super().__init__(optimizer, last_epoch)  # 初始化基类

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]  # 计算预热学习率

# 定义AverageMeter类，用于计算和记录平均值
class AverageMeter(object):
    def __init__(self):
        self.reset()  # 初始化

    def reset(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):
        self.val = val  # 更新当前值
        self.sum += val * n  # 更新总和
        self.count += n  # 更新计数
        self.avg = self.sum / self.count  # 更新平均值

# 定义accuracy函数，用于计算准确率
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)  # 获取最大的k值
    batch_size = target.size(0)  # 批次大小

    _, pred = output.topk(maxk, 1, True, True)  # 获取前k个预测值
    pred = pred.t()  # 转置预测值
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 计算预测值与真实值的匹配度

    res = []  # 用于存储每个k值的准确率
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # 计算前k个预测值的正确数量
        wrong_k = batch_size - correct_k  # 计算前k个预测值的错误数量
        res.append(wrong_k.mul_(100.0 / batch_size))  # 计算错误率并添加到列表中

    return res  # 返回每个k值的准确率

# 定义train函数，用于训练模型
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

# 设置预热周期和总训练周期
warm = 1
epoch = 160
# 定义损失函数
loss_function = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# 定义学习率调度器
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
# 计算每个周期的迭代次数
iter_per_epoch = len(trainloader)
# 创建学习率预热调度器
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

# 定义main函数，用于执行训练和验证
def main():
    # 创建训练数据集实例
    train_dataset = ProGAN_Dataset(root_dir='/opt/data/private/wangjuntong/datasets/progan_train', transform=transform)
    # 创建训练数据加载器
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=5, pin_memory=True)
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
    best_acc = 0
    # 开始训练周期
    for epoch in range(50):
        # 训练模型
        train_loss, train_acc = train(trainloader, net, loss_function, optimizer, epoch, warmup_scheduler, warm)
        # 如果结束预热，更新学习率
        if epoch >= warm:
            train_scheduler.step()

        # 打印训练结果
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}')

    # 定义待验证的数据集文件夹列表
    folders = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
    # 遍历每个文件夹进行验证
    for folder in folders:
        print(f"Validating on {folder}...")
        # 构建验证数据集路径
        val_dataset_folder = os.path.join('/opt/data/private/wangjuntong/datasets/CNN_synth_testset', folder)
        # 创建验证数据集实例
        val_dataset = ProGAN_Dataset(root_dir=val_dataset_folder, transform=transform)
        # 创建验证数据加载器
        valloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=5, pin_memory=True)

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
