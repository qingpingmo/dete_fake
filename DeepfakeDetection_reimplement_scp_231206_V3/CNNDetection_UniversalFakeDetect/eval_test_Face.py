import sys
import time
import os
import csv
import torch
from util import Logger
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
from models import get_model
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
# CUDA_VISIBLE_DEVICES=0 python eval_2.py --dataroot  ./dataset/train/ --model_path  checkpoints/20class-resnet2022_06_23_10_29_00/model_epoch_36.pth

# CUDA_VISIBLE_DEVICES=0 python eval_test_mygen.py --dataroot  ../test_onmygen_2k/ --model_path  checkpoints/20class-resnet2022_06_23_10_29_00/model_epoch_36.pth

# CUDA_VISIBLE_DEVICES=0 python eval_test_mygen.py --model_path checkpoints/20class-resnet2022_06_23_10_29_00/model_epoch_36.pth --dataroot ../test/

# vals = ['AttGAN', 'BEGAN', 'CramerGAN', 'GANimation', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
vals = ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
# multiclass = [1, 1, 1, 0, 1, 0, 0, 0]
# Running tests

vals = ['progan', 'stylegan', 'style2gan']
multiclass = [0, 0, 0]

opt = TestOptions().parse(print_options=False)

#opt.model_path = '/opt/data/private/tcc/GANS_BS1_freq/CNNDetection_UniversalFakeDetect/checkpoints/1class-resnet_horse_bs128___2023_07_22_15_20_57__lnum_64__random_sobel/model_epoch_39_89.2198.pth'
opt.model_path = 'checkpoints/4class-resnet_car_cat_chair_horse_bs128___2023_07_22_15_23_54__lnum_64__random_sobel/model_epoch_39_89.203.pth'
opt.dataroot = '/opt/data/private/tcc/data/data/FaceDataset/'
opt.loadSize  = 224


model_name = os.path.basename(opt.model_path).replace('.pth', '')

results_dir = './results_onprogan/'
logpath = os.path.join(results_dir, opt.model_path.split('/')[-2])
os.makedirs(results_dir, mode = 0o777, exist_ok = True) 
os.makedirs(logpath, mode = 0o777, exist_ok = True) 
Logger(os.path.join(logpath, opt.model_path.split('/')[-1] + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+'.log'))


dataroot = opt.dataroot
print(f'Dataroot {opt.dataroot}')
print(f'Model_path {opt.model_path}')

model = get_model('CLIP:ViT-L/14')
# model = nn.DataParallel(model)

state_dict = torch.load(opt.model_path, map_location="cpu")['model']
pretrained_dict = OrderedDict()
for ki in state_dict.keys():
    if ki.startswith('module.'):
        pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])
    # else:
    #     pretrained_dict['module.'+ki] = deepcopy(state_dict[ki])
missing, unexpected_ = model.load_state_dict(pretrained_dict)
print(f'Net_g Missing Params {missing}')
print(f'Net_g Unexpected Params {unexpected_}')

model.cuda()
model.eval()

accs = [];aps = []
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = False    # testing without resizing by default
    opt.no_crop = True    # testing without resizing by default

    # model = resnet50(num_classes=1)
    # state_dict = torch.load(opt.model_path, map_location='cpu')
    # model.load_state_dict(state_dict['model'])
    # model.cuda()
    # model.eval()
    acc, ap, _, _, _, _ = validate(model, opt)
    accs.append(acc);aps.append(ap)
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

