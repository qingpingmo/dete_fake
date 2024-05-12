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
import os
from torchvision.datasets import ImageFolder
ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root, transform):
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    
    return ImageFolder(root=root, transform=transform)


def binary_dataset(opt, root):
    fixed_height = getattr(opt, 'fixed_height', 256)
    fixed_width = getattr(opt, 'fixed_width', 256)

    transformations = [transforms.Resize((fixed_height, fixed_width))]

    if opt.isTrain:
        transformations.append(transforms.RandomCrop(opt.cropSize))
        if not opt.no_flip:
            transformations.append(transforms.RandomHorizontalFlip())

    transformations.append(transforms.Lambda(lambda img: data_augment(img, opt)))

    transformations.append(transforms.ToTensor())
    transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    dset = datasets.ImageFolder(
        root,
        transforms.Compose(transformations)
    )
    return dset








class FileNameDataset(datasets.ImageFolder):
    def __init__(self, opt, root):
        self.opt = opt
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        super(FileNameDataset, self).__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    # 确保最后返回张量
    return transforms.ToTensor()(img)



def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize, opt.loadSize), interpolation=rz_dict[interp])