#data.py
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import os
import torchvision.transforms as transforms
from .datasets import dataset_folder

vals = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def get_dataset(opt, arg_root, transform=None):
    dset_lst = []
    print("arg_root:", arg_root)  
    print("last directory name:", arg_root.split('/')[-1]) 

    if opt.isTrain:
        if 'deepfake' in arg_root or 'biggan' in arg_root or 'crn' in arg_root or 'gaugan' in arg_root or 'imle' in arg_root or 'san' in arg_root or 'seeindark' in arg_root or 'stargan' in arg_root or 'whichfaceisreal' in arg_root or 'airplane' in arg_root or 'bicycle' in arg_root or 'bird' in arg_root or 'boat' in arg_root or 'bottle' in arg_root or 'bus' in arg_root or 'car' in arg_root or 'cat' in arg_root or 'chair' in arg_root or 'cow' in arg_root or 'diningtable' in arg_root or 'dog' in arg_root or 'horse' in arg_root or 'motorbike' in arg_root or 'person' in arg_root or 'pottedplant' in arg_root or 'sheep' in arg_root or 'sofa' in arg_root or 'train' in arg_root or 'tvmonitor' in arg_root:  # 检查是否是deepfake或biggan类型的数据集
            dset = dataset_folder(opt, arg_root, transform=transform)
            dset_lst.append(dset)
        else:
            for cls in os.listdir(arg_root):
                root = arg_root + '/' + cls
                dset = dataset_folder(opt, root, transform=transform)
                dset_lst.append(dset)
    else:
        for cls in opt.classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root, transform=transform)
            dset_lst.append(dset)

    if transform is not None:
        for dset in dset_lst:
            dset.transform = transform

    return torch.utils.data.ConcatDataset(dset_lst)



def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    return sampler

def to_rgb(image):
    return image.convert('RGB')


import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

def create_dataloader(opt, root='', transform=None):
    
    transformations = [transforms.Lambda(to_rgb)] 
    transformations.append(transforms.Resize((256, 256))) 

   
    if transform is not None:
        transformations.append(transform) 

   
    combined_transforms = transforms.Compose(transformations)

    
    dataset = get_dataset(opt, root, transform=combined_transforms)
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
