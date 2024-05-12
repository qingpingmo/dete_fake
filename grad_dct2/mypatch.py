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
from torch.nn import init
from torch.optim import lr_scheduler
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# New Model Definition Starts Here
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
import torch_dct as dct
from torchvision.models import resnet

class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()
    def forward(self, x):
        x = x.view(x.size(0), int(np.prod(x.size()[1:])))
        return x

class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()
    def forward(self, x):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return x

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='max',
                                                   factor=0.1,
                                                   threshold=0.0001,
                                                   patience=opt.patience,
                                                   eps=1e-6)
    elif opt.lr_policy == 'constant':
        # dummy scheduler: min lr threshold (eps) is set as the original learning rate
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=0.1, threshold=0.0001,
                                                   patience=1000, eps=opt.lr)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='xavier', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type is None:
        return net
    init_weights(net, init_type)
    return net


class CustomResNet(nn.Module):
    '''
    Customizable ResNet, compatible with pytorch's resnet, but:
     * The top-level sequence of modules can be modified to add
       or remove or alter layers.
     * Extra outputs can be produced, to allow backprop and access
       to internal features.
     * Pooling is replaced by resizable GlobalAveragePooling so that
       any size can be input (e.g., any multiple of 32 pixels).
     * halfsize=True halves striding on the first pooling to
       set the default size to 112x112 instead of 224x224.
    '''
    def __init__(self, size=None, block=None, layers=None, num_classes=1000,
            extra_output=None, modify_sequence=None, halfsize=False):
        standard_sizes = {
            18: (resnet.BasicBlock, [2, 2, 2, 2]),
            34: (resnet.BasicBlock, [3, 4, 6, 3]),
            50: (resnet.Bottleneck, [3, 4, 6, 3]),
            101: (resnet.Bottleneck, [3, 4, 23, 3]),
            152: (resnet.Bottleneck, [3, 8, 36, 3])
        }
        assert (size in standard_sizes) == (block is None) == (layers is None)
        if size in standard_sizes:
            block, layers = standard_sizes[size]
        if modify_sequence is None:
            modify_sequence = lambda x: x
        self.inplanes = 64
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer # for recent resnet
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        sequence = modify_sequence([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2,
                padding=3, bias=False)),
            ('bn1', norm_layer(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(3, stride=1 if halfsize else 2,
                padding=1)),
            ('layer1', self._make_layer(block, 64, layers[0])),
            ('layer2', self._make_layer(block, 128, layers[1], stride=2)),
            ('layer3', self._make_layer(block, 256, layers[2], stride=2)),
            ('layer4', self._make_layer(block, 512, layers[3], stride=2)),
            ('avgpool', GlobalAveragePool2d()),
            ('fc', nn.Linear(512 * block.expansion, num_classes))
        ])
        super(CustomResNet, self).__init__()
        for name, layer in sequence:
            setattr(self, name, layer)
        self.extra_output = extra_output

    def _make_layer(self, block, channels, depth, stride=1):
        return resnet.ResNet._make_layer(self, block, channels, depth, stride)

    def forward(self, x):
        # 现在的 x 直接是 dct_images
        extra = []
        for name, module in self._modules.items():
            x = module(x)
            if self.extra_output and name in self.extra_output:
                extra.append(x)
        x = x.view(x.size(0), -1)  # Ensure the output is [batch_size, num_classes]
        if self.extra_output:
            return (x,) + tuple(extra)
        return x

def make_patch_resnet(depth, layername='layer1', num_classes=1):
    def change_out(layers):
        ind, layer = [(i, l) for i, (n, l) in enumerate(layers)
                      if n == layername][0]
        
        # Find the last instance of nn.BatchNorm2d in the layer
        bn = None
        for module in reversed(list(layer.modules())):
            if isinstance(module, nn.BatchNorm2d):
                bn = module
                break
        
        # Ensure that a BatchNorm2d layer was found
        if bn is None:
            raise RuntimeError(f"No BatchNorm2d layer found in {layername}")

        num_ch = bn.num_features  # Number of output channels from the specified layer

        # Replace layers after the specified layer with a global average pooling followed by the final convolutional layer
        layers[ind+1:] = [
            ('global_avg_pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('convout', nn.Conv2d(num_ch, num_classes, kernel_size=1))  # Ensure the number of input channels matches 'num_ch'
        ]
        return layers

    model = CustomResNet(depth, modify_sequence=change_out)
    return model








def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#    if pretrained:
       # print('+'*100)
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)

    #    model_path = '/opt/data/private/tcc/GANS_BS1_pretrainmodel/random_sobel/random_sobel_multiGPU/MyCNNDetection2_2_usingF_9_onDofFirstLayer2_addrandom_sobel_onlytest/checkpoint/4class-resnet_car_cat_chair_horse_bs128___2023_02_19_16_20_55__lnum_64__random_sobel-91.98-forpaper/model_epoch_39.pth'
     #   state_dict = torch.load(model_path, map_location='cpu')
      #  model.load_state_dict(state_dict['model'])
#        print(f"Using pretrained model{model_path}")
    # if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    model = make_patch_resnet(50, 'layer2', num_classes=2 )
    init_weights(model, init_type='xavier')
    # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)

    # from collections import OrderedDict
    # from copy import deepcopy
    # state_dict = torch.load('/opt/data/private/tcc/GANS_BS1_reimplement/MyCNNDetection2_2_patch-forensics/checkpoints/4class-resnet_car_cat_chair_horse_bs32___2023_10_22_11_26_36__lnum_64__seed_100__patch-forensics/model_epoch_Acc_82.90868015869121_epoch_86_steps391587.pth', map_location='cpu')['model']
    # net_params = sum(map(lambda x: x.numel(), model.parameters()))
    # print(f'Model parameters {net_params:,d}')
    # pretrained_dict = OrderedDict()
    # for ki in state_dict.keys():
        # pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])

    # get, miss = model.load_state_dict(pretrained_dict)#, strict=True
    
    # model.load_state_dict(torch.load('/opt/data/private/tcc/GANS_BS1_reimplement/MyCNNDetection2_2_patch-forensics/checkpoints/4class-resnet_car_cat_chair_horse_bs32___2023_10_22_11_26_36__lnum_64__seed_100__patch-forensics/model_epoch_Acc_82.90868015869121_epoch_86_steps391587.pth', map_location='cpu')['model'], strict=True)
    return model

def apply_dct(image):
    image_np = np.array(image)
    # Apply 2D DCT on the image
    dct_transformed = dct.dct_2d(torch.tensor(image_np, dtype=torch.float), norm='ortho')
    # Convert the DCT transformed tensor back to numpy for further processing
    dct_transformed_np = dct_transformed.numpy()
    dct_image = np.log(np.abs(dct_transformed_np) + 1)
    dct_image_normalized = (dct_image - np.min(dct_image)) / (np.max(dct_image) - np.min(dct_image))
    image_uint8 = (dct_image_normalized * 255).astype(np.uint8)
    return Image.fromarray(image_uint8)


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

        if self.transform:
            dct_image = self.transform(dct_image)

        label = self.labels[idx]
        return dct_image, label

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

# Use the new ResNet50 model
net = resnet50(pretrained=False)
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

def train(trainloader, model, criterion, optimizer, epoch, scheduler, warm):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (dct_images, labels) in enumerate(trainloader):
        dct_images, labels = dct_images.to(device), labels.to(device)
        
        # Print the shape of the input images and labels
        #print(f"Shape of dct_images: {dct_images.shape}")
        #print(f"Shape of labels: {labels.shape}")
        
        optimizer.zero_grad()

        outputs = model(dct_images)
        
        # Remove extra singleton dimensions from the model output
        outputs = outputs.squeeze()
        
        # Print the shape of the model output after squeezing
        #print(f"Shape of model outputs after squeezing: {outputs.shape}")

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
        for i, (dct_images, labels) in enumerate(val_loader):
            dct_images, labels = dct_images.to(device), labels.to(device)
            outputs = model(dct_images)
            outputs = outputs.squeeze()  # Ensure the output is [batch_size, num_classes]
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
    #net = ResNet18().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)
    best_acc = 0
    for epoch in range(80):
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
