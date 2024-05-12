import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
import warnings

# CUDA_VISIBLE_DEVICES=7 python eval.py --no_crop --batch_size 1 --model_path ./checkpoints/blur_jpg_prob0.1_1class_horse/model_epoch_latest.pth

warnings.filterwarnings('ignore')

# Running tests
opt = TestOptions().parse(print_options=False)
model_path = opt.model_path
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = False    # testing without resizing by default

    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, _, _, _, _ = validate(model, opt)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
