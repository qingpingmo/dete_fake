import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from sklearn.metrics import average_precision_score
import os
from scipy.fftpack import idct
import torch_dct as dct

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, lnum=1):
        super(ResNet, self).__init__()
        
        self.printOne = 1
        self.lnum = lnum*2-1
        kshape = [lnum, 1]
        self.kshape = kshape
        mean = np.load('GANDCTAnalysis/saved_mean_var/mean.npy')  # 256, 256, 3
        var  = np.load('GANDCTAnalysis/saved_mean_var/var.npy')
        
        self.mean_w = nn.Parameter( torch.from_numpy( mean), requires_grad=False).detach().cuda().permute(2,0,1).squeeze(0).float()  # 1,3,256,256
        self.var_w  = nn.Parameter( torch.from_numpy( var ), requires_grad=False).detach().cuda().permute(2,0,1).squeeze(0).float()  # 1,3,256,256
        
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])            # 256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 512
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 1024
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Adjust the final fully connected layer to output two values per instance
        self.fc1 = nn.Linear(512 * block.expansion, 2)  # Change from 1 to 2

        # self.fc1 = nn.Linear(512, 1)

        #for m in self.modules():
        #    print(m)
        #exit()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
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
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
    def forward(self, x):

        x = dct.dct_2d(x, norm='ortho')
        x = torch.abs(x)
        x += 1e-13
        x = torch.log(x)

        # remove mean + unit variance
        x = x - self.mean_w
        x = x / self.var_w
        # tmp = x; print(f'x shape: {tmp.shape}, max: {tmp.max()}, min: {tmp.min()}')
        
        x = self.conv1( x ) #64
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)# in64 out256
        x = self.layer2(x)# in256  out512
        x = self.layer3(x)# in512  out1024
        x = self.layer4(x)# in1024 out2048

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

# Dataset class for loading data
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
            else:
                label = 1 if '1_fake' in path else 0
                self.image_paths.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transformations to the original image
        if self.transform:
            transformed_image = self.transform(image)

        # Convert image to PyTorch tensor and apply DCT transformation using torch_dct
        image_tensor = transforms.ToTensor()(image)  # Convert PIL image to tensor
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_dct = dct.dct_2d(image_tensor, norm='ortho')  # Apply 2D DCT
        image_dct = torch.abs(image_dct)  # Take the absolute value to ensure non-negative values
        image_dct_log = torch.log(image_dct + 1e-5)  # Apply log transformation

        # Normalize DCT data to [0, 1] range
        image_dct_norm = (image_dct_log - image_dct_log.min()) / (image_dct_log.max() - image_dct_log.min())
        image_dct_norm = image_dct_norm.squeeze(0)  # Remove batch dimension

        # If transformations are specified, apply them to the DCT-transformed image
        if self.transform:
            transformed_dct_image = self.transform(transforms.ToPILImage()(image_dct_norm))  # Convert tensor to PIL image and apply transform

        label = self.labels[idx]
        return transformed_dct_image, label




# Transformation for input data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

warm = 1

# Initialize the model
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

train_dataset = ProGAN_Dataset(root_dir='/opt/data/private/wangjuntong/datasets/progan_train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

val_dataset = ProGAN_Dataset(root_dir='/opt/data/private/wangjuntong/datasets/CNN_synth_testset/deepfake', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Scheduler and WarmUpLR
scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
total_iters = len(train_loader)

class WarmUpLR(StepLR):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, step_size=1, gamma=0.1, last_epoch=last_epoch)

    def get_lr(self):
        return [base_lr * min(1.0, self.last_epoch / self.total_iters) for base_lr in self.base_lrs]

warmup_scheduler = WarmUpLR(optimizer, total_iters * warm)

# Training function with warmup
def train(epoch, warmup_scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if warmup_scheduler is not None and epoch <= warm:
            warmup_scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f'Train Epoch: {epoch} Loss: {running_loss / len(train_loader):.6f}, Acc: {100. * correct / total:.2f}%')

def validate(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []

    with torch.no_grad():  # No need to track gradients for validation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Extend the lists for average precision calculation
            all_predicted.extend(predicted.view(-1).cpu().numpy())
            all_targets.extend(targets.view(-1).cpu().numpy())

    # Calculate validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    # Calculate the average precision score
    val_ap = average_precision_score(all_targets, all_predicted)

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%, Validation AP: {val_ap:.4f}')

    return val_loss, val_acc, val_ap

# Main training loop with warm-up and scheduler
epochs = 160  # Total number of epochs including warm-up
warm = 5  # Number of warm-up epochs

for epoch in range(1, epochs + 1):
    if epoch > warm:
        scheduler.step()
    train(epoch, warmup_scheduler if epoch <= warm else None)
    validate(model, val_loader, criterion)  # Corrected this line