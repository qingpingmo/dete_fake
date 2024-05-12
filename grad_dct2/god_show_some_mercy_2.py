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
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from sklearn.metrics import average_precision_score
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo




os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


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



class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    return model

def return_pytorch04_xception(pretrained=True):
    model = xception(pretrained=False)
    if False:
        state_dict = torch.load('xception-b5690688.pth', map_location='cpu')
        # state_dict = model_zoo.load_url(pretrained_settings['xception']['imagenet']['url'],map_location='cpu')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict, strict=False)
 
    return model

class Filter(nn.Module):
    def __init__(self, size, 
                 band_start, 
                 band_end, 
                 use_learnable=True, 
                 norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class FAD_Head(nn.Module):
    def __init__(self, size=256):  # 默认值设置为256
        super(FAD_Head, self).__init__()
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.tensor(DCT_mat(size)).float().transpose(0, 1), requires_grad=False)
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        print(f"Input x shape: {x.shape}")  # [B, C, H, W]
        batch_size, channels, height, width = x.shape

        # 使用DCT矩阵执行2D DCT变换
        x_freq = torch.einsum('bhwc,cm,nm->bhwn', x, self._DCT_all, self._DCT_all_T)
        print(f"x_freq shape after DCT: {x_freq.shape}")

        y_list = []
        for filter in self.filters:
            x_filtered = filter(x_freq)
            # 对滤波后的结果执行逆DCT变换
            y = torch.einsum('bhwn,nc,mc->bhwc', x_filtered, self._DCT_all_T, self._DCT_all)
            y_list.append(y)

        out = torch.cat(y_list, dim=1)
        print(f"Output shape: {out.shape}")
        return out




class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)

        # init filters
        self.filters = nn.ModuleList([Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i+1), norm=True) for i in range(M)])
    
    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8)/2) + 1
        assert size_after == 149

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)   # [N, C * S * S, L]   L:block num
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        # M kernels filtering
        y_list = []
        for i in range(self._M):
            # y = self.filters[i](x_dct)    # [N, L, C, S, S]
            # y = torch.abs(y)
            # y = torch.sum(y, dim=[2,3,4])   # [N, L]
            # y = torch.log10(y + 1e-15)
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2,3,4])
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)   # [N, 1, 149, 149]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
        return out
    
class MixBlock(nn.Module):
    
    def __init__(self, c_in, width, height):
        super(MixBlock, self).__init__()
        self.FAD_query = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1,1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1,1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)    # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)    # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  #[BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))
        return y_FAD, y_LFS

# utils
def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return np.array(m)


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.



class F3Net(nn.Module):
    """
    Implementation is mainly referenced from https://github.com/yyk-wew/F3Net
    """
    def __init__(self, 
                 num_classes: int=2, 
                 img_width: int=299, 
                 img_height: int=299, 
                 LFS_window_size: int=10, 
                 LFS_M: int=6) -> None:
        super(F3Net, self).__init__()
        assert img_width == img_height
        self.img_size = img_width
        self.num_classes = num_classes
        self._LFS_window_size = LFS_window_size
        self._LFS_M = LFS_M
        
        
        self.fad_head = FAD_Head(self.img_size)
        self.lfs_head = LFS_Head(self.img_size, self._LFS_window_size, self._LFS_M)
        
        self.fad_excep = self._init_xcep_fad()
        self.lfs_excep = self._init_xcep_lfs()
        
        self.mix_block7 = MixBlock(c_in=728, width=19, height=19) 
        self.mix_block12 = MixBlock(c_in=1024, width=10, height=10) 
        self.excep_forwards = ['conv1', 'bn1', 'relu', 'conv2', 'bn2', 'relu', 
                               'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 
                               'block7', 'block8', 'block9', 'block10' , 'block11', 'block12',
                               'conv3', 'bn3', 'relu', 'conv4', 'bn4']

         # classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096, num_classes)
        self.dp = nn.Dropout(p=0.2)
        
    def _init_xcep_fad(self):
        fad_excep =  return_pytorch04_xception(True)
        conv1_data = fad_excep.conv1.weight.data
        # let new conv1 use old param to balance the network
        fad_excep.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
            fad_excep.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0
        return fad_excep
    
    def  _init_xcep_lfs(self): 
        lfs_excep = return_pytorch04_xception(True)
        conv1_data = lfs_excep.conv1.weight.data
        # let new conv1 use old param to balance the network
        lfs_excep.conv1 = nn.Conv2d(self._LFS_M, 32, 3, 1, 0, bias=False)
        for i in range(int(self._LFS_M / 3)):
            lfs_excep.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / float(self._LFS_M / 3.0)
        return lfs_excep
    
    def _features(self, x_fad, x_fls):
        for forward_func in self.excep_forwards:
            x_fad = getattr(self.fad_excep, forward_func)(x_fad)
            x_fls = getattr(self.lfs_excep, forward_func)(x_fls)
            if forward_func == 'block7':
                x_fad, x_fls = self.mix_block7(x_fad, x_fls)
            if forward_func == 'block12':
                x_fad, x_fls = self.mix_block12(x_fad, x_fls)
        return x_fad, x_fls
    
    def _norm_feature(self, x):
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x):
        fad_input = self.fad_head(x)
        print(f"fad_input shape: {fad_input.shape}") 
        lfs_input = self.lfs_head(x)
        print(f"lfs_input shape: {lfs_input.shape}") 
        x_fad, x_fls = self._features(fad_input, lfs_input)
        x_fad = self._norm_feature(x_fad)
        x_fls = self._norm_feature(x_fls)
        x_cat = torch.cat((x_fad, x_fls), dim=1)
        x_drop = self.dp(x_cat)
        logit = self.fc(x_drop)
        #return x_cat, logit
        return logit


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

class FusionCapsNetWithF3Net(nn.Module):
    def __init__(self):
        super(FusionCapsNetWithF3Net, self).__init__()
        # Initialize F3Net
        
        self.f3net = F3Net(num_classes=2, img_width=299, img_height=299, LFS_window_size=10, LFS_M=6)
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

        self.feature_fusion = nn.Sequential(
            # Define your feature fusion layers here
            nn.Conv2d(256 * 3, 512, kernel_size=1),  # Example fusion layer
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

        self.classifier = nn.Linear(288000, 2)  # Adjust the input size based on the output of the capsule layer

    def forward(self, x_dct, x_grad):
        print(f"x_dct shape: {x_dct.shape}")  # 打印x_dct的形状
        print(f"x_grad shape: {x_grad.shape}")  # 打印x_grad的形状
        print(f"x_dct shape entering F3Net: {x_dct.shape}")
        
        # Extract F3Net features
        f3net_features = self.f3net(x_dct)  # You may need to adjust this call based on the actual input requirements of F3Net

        # Extract DCT and Gradient features as before
        dct_features = self.dct_feature_extractor(x_dct)
        grad_features = self.grad_feature_extractor(x_grad)

        # Combine F3Net features with DCT and Gradient features
        combined_features = torch.cat([f3net_features, dct_features, grad_features], dim=1)
        combined_features = self.feature_fusion(combined_features)

        # Proceed as in FusionCapsNet
        combined_features = self.cbam(combined_features)
        capsule_output = self.capsule_layer(combined_features)
        capsule_output = capsule_output.view(x_dct.size(0), -1)
        outputs = self.classifier(capsule_output)

        return outputs



net = FusionCapsNetWithF3Net()
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
    net = FusionCapsNetWithF3Net().to(device)
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