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



vals = ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Running tests
opt = TestOptions().parse(print_options=False)


opt.model_path = 'checkpoints/4class-resnet_car_cat_chair_horse_bs128___2023_07_22_15_23_54/model_epoch_39_89.203.pth'
# opt.dataroot = '/opt/data/private/tcc/data/data/CNNDetection/test_onmygen_2k/'
opt.loadSize  = 224
opt.batch_size = 32


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

    acc, ap, _, _, _, _ = validate(model, opt)
    accs.append(acc);aps.append(ap)
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

