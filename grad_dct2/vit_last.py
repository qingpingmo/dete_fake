from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
np.object = object 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.efficient import ViT
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Training settings
batch_size = 128
epochs = 200
lr = 3e-5
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

import os
import glob
from PIL import Image
from torch.utils.data import Dataset

train_dir = '/opt/data/private/wangjuntong/datasets/progan_train'
val_dir = '/opt/data/private/wangjuntong/datasets/CNN_synth_testset'
train_image_paths = []  # Training images
val_image_paths = []  # Validation images
train_labels = []  # Labels for training images
val_labels = []  # Labels for validation images


def train_load_images(train_dir):
    if 'new_features' in train_dir.split(os.sep):
        return
    for item in os.listdir(train_dir):  # 遍历当前目录
        path = os.path.join(train_dir, item)  # 获取文件或目录的完整路径
        if os.path.isdir(path):  # 如果是目录
            train_load_images(path)  # 递归调用加载图像
        elif path.endswith('.png'):  # 如果是PNG图像文件
            label = 1 if '1_fake' in path else 0  # 根据文件名判断图像是否为假图像，并设置标签
            train_image_paths.append(path)  # 添加图像路径到列表
            train_labels.append(label)  # 添加标签到列表



def val_load_images(val_dir):
    if 'new_features' in val_dir.split(os.sep):
        return
    for item in os.listdir(val_dir):  # 遍历当前目录
        path = os.path.join(val_dir, item)  # 获取文件或目录的完整路径
        if os.path.isdir(path):  # 如果是目录
            val_load_images(path)  # 递归调用加载图像
        elif path.endswith('.png'):  # 如果是PNG图像文件
            label = 1 if '1_fake' in path else 0  # 根据文件名判断图像是否为假图像，并设置标签
            val_image_paths.append(path)  # 添加图像路径到列表
            val_labels.append(label)  # 添加标签到列表



train_load_images('/opt/data/private/wangjuntong/datasets/progan_train')
val_load_images('/opt/data/private/wangjuntong/datasets/CNN_synth_testset')
train_list=train_image_paths
test_list=val_image_paths

train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=train_labels,
                                          random_state=seed)

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, labels,transform=None):
        self.file_list = file_list
        self.transform = transform
        self.labels=labels

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label=self.labels[idx]
        return img_transformed, label


train_data = CatsDogsDataset(train_list, train_labels,transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, train_labels,transform=test_transforms)
test_data = CatsDogsDataset(test_list,val_labels, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)


# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )



