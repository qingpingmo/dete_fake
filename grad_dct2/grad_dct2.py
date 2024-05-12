import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.utils.model_zoo as model_zoo
from PIL import Image
from scipy.fftpack import dct
from torchvision import transforms
import scipy.fftpack as fftpack
# Specify the subfolders you are interested in
SUBFOLDERS = ['cat', 'car', 'horse', 'chair']

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Helper function to load mean and variance files
def load_mean_var(mean_path, var_path):
    if not os.path.exists(mean_path) or not os.path.exists(var_path):
        raise FileNotFoundError(f"Required files not found: {mean_path}, {var_path}. Please check their existence and path.")
    mean = np.load(mean_path)
    var = np.load(var_path)
    print(f"Mean shape: {mean.shape}, Var shape: {var.shape}")  # 打印形状
    return mean, var


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



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)



        out += identity
        out = self.relu(out)

        return out


def resize_mean_var(array, new_size):
    """调整 mean 或 var 的尺寸以匹配新的图像尺寸"""
    if array.ndim == 3 and array.shape[2] == 3:  # 如果输入是三维且有三个通道
        array = array.mean(axis=2)  # 计算通道均值
    tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)  # 转换为 4D 张量 (1 x 1 x H x W)
    resized_tensor = F.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)
    return resized_tensor.squeeze(0).numpy()  # 移除批量维度，返回 2D 张量 (1 x H x W)




class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, lnum=1, input_channels=1):
        super(ResNet, self).__init__()
        self.lnum = lnum*2-1
        kshape = [lnum, 1]
        self.kshape = kshape

        # Load mean and variance
        mean_path = 'GANDCTAnalysis/saved_mean_var/mean.npy'
        var_path = 'GANDCTAnalysis/saved_mean_var/var.npy'
        mean, var = load_mean_var(mean_path, var_path)
        mean_resized = resize_mean_var(mean, new_size=(64, 64))
        var_resized = resize_mean_var(var, new_size=(64, 64))

        self.mean_w = nn.Parameter(torch.from_numpy(mean_resized), requires_grad=False).detach().cuda().permute(2,0,1).squeeze(0).float()
        self.var_w  = nn.Parameter(torch.from_numpy(var_resized), requires_grad=False).detach().cuda().permute(2,0,1).squeeze(0).float()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])            
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) 

        # 添加 DANet 头部
        self.da_head = _DAHead(in_channels=512 * block.expansion, nclass=num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1000, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # DCT 变换
        #x_np = x.detach().cpu().numpy()
        #x_dct = fftpack.dct(x_np, axis=2, norm='ortho')
        #x_dct = fftpack.dct(x_dct, axis=3, norm='ortho')
        #x = torch.from_numpy(x_dct).to(x.device)

        #if x.shape[1] != 1:
        #    x = x.mean(dim=1, keepdim=True)

        #x = torch.abs(x)
        #x += 1e-13
        #x = torch.log(x)

        # Remove mean and apply unit variance
        #x = x - self.mean_w
        #x = x / self.var_w
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 使用 DANet
        x = self.da_head(x)

        # 选择 _DAHead 输出的第一个元素
        if isinstance(x, tuple):
            #print("Output dimensions from _DAHead:")
            #for output in x:
                #print(output.size())
            x = x[0]
        #else:
            #print("Single output dimensions from _DAHead:", x.size())

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print("Dimensions before entering the fc layer:", x.size())
        x = self.fc(x)
        #print(x.size)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

# Load images from specific folders
def load_images_from_folder(base_folder, image_size=(64, 64)):
    images = []
    labels = []
    for folder in SUBFOLDERS:
        class_folder = os.path.join(base_folder, folder)
        if os.path.isdir(class_folder):
            for subfolder in ['0_real', '1_fake']:
                subfolder_path = os.path.join(class_folder, subfolder)
                label = 1 if subfolder == '1_fake' else 0
                for filename in os.listdir(subfolder_path):
                    if filename.endswith(".png"):
                        img_path = os.path.join(subfolder_path, filename)
                        with Image.open(img_path) as img:
                            img = img.convert('L')  # 转换为灰度图像
                            img = img.resize(image_size)
                            images.append(img.copy())
                            labels.append(label)
    return images, labels


# Perform DCT on images
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def process_images_with_dct(images):
    dct_images = []
    for img in images:
        gray = img.convert('L')
        gray_array = np.array(gray)
        dct_transformed = dct2(gray_array)
        dct_images.append(dct_transformed)
    return dct_images


# Define AttentionNet
class AttentionNet(nn.Module):
    def __init__(self, image_size):
        super(AttentionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.image_size = image_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x

# Apply attention to DCT images
def apply_attention_to_dct_images(dct_images, trained_attention_net, device, batch_size=32):
    processed_images = []
    for i in range(0, len(dct_images), batch_size):
        batch_dct_images = dct_images[i:i+batch_size]
        batch_dct_images_tensor = torch.stack([torch.from_numpy(dct_image).float().unsqueeze(0) for dct_image in batch_dct_images])
        batch_dct_images_tensor = batch_dct_images_tensor.to(device)
        attention_map = trained_attention_net(batch_dct_images_tensor)
        dct_prime = batch_dct_images_tensor * attention_map
        processed_images.extend(dct_prime.detach().cpu().numpy())
    return processed_images



def apply_attention_to_images(images, trained_attention_net, device, batch_size=32):
    processed_images = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_images_tensor = torch.stack([transforms.ToTensor()(image) for image in batch_images])
        batch_images_tensor = batch_images_tensor.to(device)

        # 确保输入维度正确
        if batch_images_tensor.ndim == 5:
            batch_images_tensor = torch.squeeze(batch_images_tensor, 1)

        attention_map = trained_attention_net(batch_images_tensor)
        processed_images.extend(attention_map.detach().cpu().numpy())
    return processed_images


# Save processed images
def save_processed_images(processed_images, labels, save_path):
    with open(save_path, 'wb') as file:
        pickle.dump((processed_images, labels), file)

# Load processed images
def load_processed_images(load_path):
    with open(load_path, 'rb') as file:
        return pickle.load(file)

# ImageDataset class
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

def initialize_custom_resnet(input_channels=64, num_classes=2):
    # Load the basic ResNet model
    resnet = resnet18(pretrained=False, input_channels=input_channels)
    
    # Load pretrained weights, but ignore the final layer
    pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
    model_dict = resnet.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrained_dict)
    resnet.load_state_dict(model_dict)

    # Replace the final layer to match the desired number of classes
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

# Train model
def train_model(model, criterion, optimizer, dataloader, epochs=10, device=None):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(dataloader)
        for i, (data, target) in enumerate(dataloader, 1):
            # 确保数据是单通道的
            if data.shape[1] != 1:
                data = data.mean(dim=1, keepdim=True)
            data, target = data.to(device), target.to(device)
            # 现在data的shape应该是[batch_size, 1, height, width]
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {i}/{total_batches}, Batch Loss: {loss.item()}')
        epoch_loss = running_loss / total_batches
        print(f'End of Epoch {epoch + 1}, Average Loss: {epoch_loss}')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    folder_path = '/root/rstao/datasets/progan_train'
    all_images, all_labels = load_images_from_folder(folder_path, image_size=(64, 64))

    # 不再进行 DCT 变换
    trained_attention_net = AttentionNet(image_size=64).to(device)
    torch.cuda.empty_cache()
    # 直接处理原始图像
    processed_images = apply_attention_to_images(all_images, trained_attention_net, device)

    resnet = initialize_custom_resnet(input_channels=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(resnet.parameters(), lr=0.01)
    dataset = ImageDataset(processed_images, all_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    train_model(resnet, criterion, optimizer, dataloader, device=device)

if __name__ == '__main__':
    main()



