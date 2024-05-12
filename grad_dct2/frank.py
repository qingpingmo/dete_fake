import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
#from tensorboardX import SummaryWriter
import numpy as np
#from validate import validate
#from data import create_dataloader
#from earlystop import EarlyStopping
#from networks.trainer import Trainer
#from options.train_options import TrainOptions
#from options.test_options import TestOptions
# from eval_config import *
#from util import Logger
#import util
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
import torch_dct as dct
import torch
import numpy as np
#from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
#from options.test_options import TestOptions
#from data import create_dataloader

import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode
import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import functools
import torch
import torch.nn as nn
#from networks.base_model import BaseModel, init_weights

#from .datasets import dataset_folder
import random


class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='res50', help='architecture for binary classification')

        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0)
        parser.add_argument('--blur_sig', default='0.5')
        parser.add_argument('--jpg_prob', type=float, default=0)
        parser.add_argument('--jpg_method', default='cv2')
        parser.add_argument('--jpg_qual', default='75')

        parser.add_argument('--dataroot', default='./dataset/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--classes', default='', help='image classes to train on')
        parser.add_argument('--class_bal', action='store_true')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--delr_freq', type=int, default=20, help='frequency of change lr')
        parser.add_argument('--test_freq', type=int, default=4000, help='frequency of change lr')
        parser.add_argument('--delr', type=float, default=0.8, help='delr')
        parser.add_argument('--pth', type=str, default='karras2019stylegan-bedrooms-256x256_discriminator.pth', help='frequency of change lr')
        parser.add_argument('--lnum', type=int, default=1, help='fixed num layer')
        parser.add_argument('--seed', type=int, default=123, help='fixed num layer')

        
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return opt
        # return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        try:
            thePth = opt.pth.split('/')[-1].split('.')[0]
        except:
            thePth = opt.pth.replace('/','_').replace('.','_')

        opt.name = '__'.join([opt.name, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'lnum_'+str(opt.lnum), 'seed_'+str(opt.seed),thePth])
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        opt.classes = opt.classes.split(',')
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--loss_freq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        # parser.add_argument('--model_path')
        # parser.add_argument('--no_resize', action='store_true')
        # parser.add_argument('--no_crop', action='store_true')
        self.isTrain = True
        return parser



class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--dataroot')
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')

        self.isTrain = False
        return parser



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=1, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_max = -np.Inf
        self.delta = delta

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.score_max:.6f} --> {score:.6f}).  Saving model ...')
        model.save_networks('best')
        self.score_max = score


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                # transforms.Resize((256, 256)),
                # transforms.Lambda(lambda img: data_augment(img, opt)),
                # crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset

class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    # print(dset_lst)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler

def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt








class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    def save_networks(self, epoch):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
        }

        torch.save(state_dict, save_path)

    # load models from the disk
    def load_networks(self, epoch):
        load_filename = 'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)

        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.model.load_state_dict(state_dict['model'])
        self.total_steps = state_dict['total_steps']

        if self.isTrain and not self.opt.new_optim:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            ### move optimizer state to GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.optimizer.param_groups:
                g['lr'] = self.opt.lr

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type='normal', gain=0.02):
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
        mean = np.load('../mean.npy')  # 256, 256, 3
        var  = np.load('../var.npy')
        
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
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model










class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.delr = opt.delr

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=False, lnum=opt.lnum)
            # self.model.fc = nn.Linear(2048, 1)
            # torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)
        print('='*30)
        for name,pa in self.model.named_parameters():
            if pa.requires_grad: print('='*20, 'requires_grad Ture',name)
        print('='*30)
        for name,pa in self.model.named_parameters():
            if not pa.requires_grad: print('='*20, 'requires_grad False',name)
        print('='*30)
        net_params = sum(map(lambda x: x.numel(), self.model.parameters()))
        print(f'Model parameters {net_params:,d}')
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                # self.optimizer = torch.optim.Adam(self.model.parameters(),
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        # self.model.to(opt.gpu_ids[0])
        self.model = nn.DataParallel(self.model).cuda()


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.delr
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/self.delr} to {param_group["lr"]} with delr {self.delr}')
        print('*'*25)
        return True

    def set_input(self, input):
        # self.input = input[0].to(self.device)
        # self.label = input[1].to(self.device).float()
        self.input = input[0].cuda()
        self.label = input[1].cuda().float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(opt.seed)
    dataroot = os.path.join(opt.dataroot,'test')
    print(dataroot)
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    cmd = ' '.join(sys.argv)
    cwd = os.getcwd()
    print(cmd)
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)
    
    def testmodel():
        print(f'cwd: {cwd}')
        print(f'cmd: {cmd}')
        print('*'*25);accs = [];aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(dataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.no_resize = False    # testing without resizing by default
            Testopt.no_crop = True    # testing without resizing by default
            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc);aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        return np.array(accs).mean()*100
    model.eval();testmodel();model.train()
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            # if model.total_steps % opt.test_freq == 0:
            #     model.eval();
            #     test_tmp = testmodel()
            #     if  test_tmp > 92.:
            #         model.save_networks(f'epoch_{epoch}_steps{model.total_steps}_{test_tmp}')
            #     model.train()
            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()
            

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        test_tmp = testmodel()
        model.save_networks(f'Acc_{test_tmp}_epoch_{epoch}_steps{model.total_steps}')
        print(os.getcwd())
        print('saving the latest model %s (epoch %d, model.total_steps %d)' %(opt.name, epoch, model.total_steps))
        # testmodel()


        model.train()

