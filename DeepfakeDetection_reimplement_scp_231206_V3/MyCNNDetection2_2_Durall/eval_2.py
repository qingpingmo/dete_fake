import sys
import time
import os
import csv
import torch
from util import Logger
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
import networks.resnet as resnet
import numpy as np


# CUDA_VISIBLE_DEVICES=0 python eval_2.py --dataroot  ./dataset/train/ --model_path  checkpoints/1class-resnet_horse/model_epoch_latest.pth
# CUDA_VISIBLE_DEVICES=0 python eval.py --dataroot  ./dataset/test/ --model_path  checkpoints/1class-resnet_horse/model_epoch_latest.pth

# CUDA_VISIBLE_DEVICES=0 python eval_2.py --dataroot  /root/train/ --model_path  checkpoints/4class-resnet_car_cat_chair_horse_bs128___2023_01_03_12_23_05__lnum_1__karras2019stylegan-bedrooms-256x256_discriminator/model_epoch_99.pth --batch_size 128

vals = ['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(opt.model_path).replace('.pth', '')

results_dir = './results_onprogan/'
logpath = os.path.join(results_dir, opt.model_path.split('/')[-2])
os.makedirs(results_dir, mode = 0o777, exist_ok = True) 
os.makedirs(logpath, mode = 0o777, exist_ok = True) 
Logger(os.path.join(logpath, opt.model_path.split('/')[-1] + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+'.log'))

dataroot = opt.dataroot
print(f'Dataroot {opt.dataroot}')
print(f'Model_path {opt.model_path}')

model = resnet50(num_classes=1)
state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
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
    print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id, val, acc*100, ap*100))
print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

