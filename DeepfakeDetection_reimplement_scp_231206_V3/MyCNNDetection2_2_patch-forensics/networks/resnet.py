import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
import torch_dct as dct
from torchvision.models import resnet
from networks import netutils


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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
        self.fc1 = nn.Linear(512 * block.expansion, 1)
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



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


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
        extra = []
        for name, module in self._modules.items():
            x = module(x)
            if self.extra_output and name in self.extra_output:
                extra.append(x)
        if self.extra_output:
            return (x,) + tuple(extra)
        return x

def make_patch_resnet(depth, layername='layer1', num_classes=1):
    def change_out(layers):
        ind, layer = [(i, l) for i, (n, l) in enumerate(layers)
                      if n == layername][0]
        if layername.startswith('layer'):
            bn = list(layer.modules())[-1 if depth < 50 else -2] # find final batchnorm
            assert(isinstance(bn, nn.BatchNorm2d))
            num_ch = bn.num_features
        else:
            num_ch = 64
        layers[ind+1:] = [('convout', nn.Conv2d(num_ch, num_classes, kernel_size=1))]
        return layers
    model = CustomResNet(depth, modify_sequence=change_out)
    return model


class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()
    def forward(self, x):
        x = x.view(x.size(0), int(numpy.prod(x.size()[1:])))
        return x

class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()
    def forward(self, x):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return x


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
    netutils.init_weights(model, init_type='xavier')
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



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
