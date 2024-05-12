import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np



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
        # self.weight1 = nn.Parameter( torch.randn((kshape[0], 3, kshape[1], kshape[1])) ,requires_grad=False)# .cuda() 
        # self.bias1 = nn.Parameter(torch.randn((kshape[0],)) ,requires_grad=False) #.cuda() 
        # print(f'self.weight1.shape: {self.weight1.shape}')
        
        # channel = self.weight1.shape[0]
        # kernel_v = np.array([[0, -1, 0],
        #                     [ 0,  0, 0],
        #                     [ 0,  1, 0]]).reshape(3, 3)
        # kernel_h = np.array([[0, 0, 0],
        #                     [-1, 0, 1],
        #                     [ 0, 0, 0]]).reshape(3, 3)
        # sobelv = np.zeros([channel,channel,3,3])
        # sobelh = np.zeros([channel,channel,3,3])
        # for index in range(channel):
        #     sobelv[index, index,:,:] = kernel_v
        #     sobelh[index, index,:,:] = kernel_h
        # self.sobelv = nn.Parameter( torch.Tensor(sobelv) ,requires_grad=False).detach().cuda() #
        # self.sobelh = nn.Parameter( torch.Tensor(sobelh) ,requires_grad=False).detach().cuda() #


        # print('+'*15)
        # print(self.sobelv.requires_grad)
        # print(self.sobelh.requires_grad)
        # print(self.weight1.requires_grad)
        # print(self.bias1.requires_grad)
        # print('+'*15)

        # self.weight1.requires_grad         = False
        # self.bias1.requires_grad           = False   
        # self.sobelv.requires_grad          = False
        # self.sobelh.requires_grad          = False

        # print('+'*15)
        # print(self.sobelv.requires_grad)
        # print(self.sobelh.requires_grad)
        # print(self.weight1.requires_grad)
        # print(self.bias1.requires_grad)
        # print('+'*15)
        
        self.inplanes = 64
        
        # self.conv1_2 = nn.Conv2d(self.weight1.shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])            # 256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 512
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 1024
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, 1)
        self.fc1 = nn.Linear(512, 1)

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
        # if self.printOne <10:
        #     print(self.weight1[0,:,0,0].clone().detach().cpu().numpy().tolist())
        #     self.printOne += 1
        # x = F.conv2d(x, self.weight1, self.bias1, stride=1, padding=0)
        # x = F.relu(x, inplace=True)

        # x_v = F.conv2d(x, self.sobelv, stride=1, padding=1)
        # x_h = F.conv2d(x, self.sobelh, stride=1, padding=1)
        # x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6).clone().detach()

        # x = self.conv1_2(x) #64
        # x200 = x - self.interpolate(x, 2)
        # x150 = x - self.interpolate(x, 1.50)
        factor = 0.5
        x_half = F.interpolate(x, scale_factor=factor, mode='nearest', recompute_scale_factor=True)
        x_re   = F.interpolate(x_half, scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
        x50  = x - x_re

        # ind = 6
        # print('x/n',x[0,0,:ind,:ind])
        # print('x_half/n', x_half[0,0,:ind//2,:ind//2])
        # print('x_re/n', x_re[0,0,:ind,:ind])
        # print('x50/n',x50[0,0,:ind,:ind])


        x = self.conv1( x50*2./3. ) #64
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)# in64 out256
        x = self.layer2(x)# in256  out512
        # x = self.layer3(x)# in512  out1024
        # x = self.layer4(x)# in1024 out2048

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


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#    if pretrained:
       # print('+'*100)
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)

    #    model_path = '/opt/data/private/tcc/GANS_BS1_pretrainmodel/random_sobel/random_sobel_multiGPU/MyCNNDetection2_2_usingF_9_onDofFirstLayer2_addrandom_sobel_onlytest/checkpoint/4class-resnet_car_cat_chair_horse_bs128___2023_02_19_16_20_55__lnum_64__random_sobel-91.98-forpaper/model_epoch_39.pth'
     #   state_dict = torch.load(model_path, map_location='cpu')
      #  model.load_state_dict(state_dict['model'])
#        print(f"Using pretrained model{model_path}")
    # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
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
