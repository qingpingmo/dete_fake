import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
import torch_dct as dct
from torchvision.models import resnet
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
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