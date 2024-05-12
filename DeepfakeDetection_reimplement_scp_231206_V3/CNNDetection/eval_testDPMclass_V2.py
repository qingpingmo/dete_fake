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
import warnings
warnings.filterwarnings("ignore")


vals = ['DALLE', 'guided-diffusion',  'improved-diffusion',  'midjourney', 'ddpm'] 
multiclass = [0, 0, 0, 0, 1]
# Running tests
opt = TestOptions().parse(print_options=False)
# opt.dataroot = "/opt/data/private/tcc/GANs/DM/testDP/"

opt.model_path = 'checkpoints/blur_jpg_prob0.1_4class_1282022_12_28_16_31_58/model_epoch_99.pth'
opt.batch_size = 1


model_name = os.path.basename(opt.model_path).replace('.pth', '')

results_dir = './results_onprogan/'
logpath = os.path.join(results_dir, opt.model_path.split('/')[-2])
os.makedirs(results_dir, mode = 0o777, exist_ok = True) 
os.makedirs(logpath, mode = 0o777, exist_ok = True) 
Logger(os.path.join(logpath, opt.model_path.split('/')[-1] + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+'.log'))

# print(sys.argv)
# print()
dataroot = opt.dataroot
print(f'Dataroot {opt.dataroot}')
print(f'Model_path {opt.model_path}')

accs = [];aps = []
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True    # testing without resizing by default
    opt.no_crop = True    # testing without resizing by default

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()
    acc, ap, _, _, _, _ = validate(model, opt)
    accs.append(acc);aps.append(ap)
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

