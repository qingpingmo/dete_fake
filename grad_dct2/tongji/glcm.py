import cv2
import numpy as np
import os

# 定义最大灰度级数
gray_level = 16

def maxGrayLevel(img):
    return np.max(img) + 1

def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = np.zeros((gray_level, gray_level))
    height, width = srcdata.shape
    
    max_gray_level = maxGrayLevel(input)
    
    if max_gray_level > gray_level:
        srcdata = (srcdata / (max_gray_level / gray_level)).astype(int)
 
    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1
 
    ret = ret / float(height * width)
    return ret

def feature_computer(p):
    Con, Eng, Asm, Idm = 0.0, 0.0, 0.0, 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) ** 2 * p[i][j]
            Asm += p[i][j] ** 2
            Idm += p[i][j] / (1 + (i - j) ** 2)
            if p[i][j] > 0:
                Eng += p[i][j] * np.log(p[i][j])
    return Asm, Con, -Eng, Idm

def process_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_0 = getGlcm(img_gray, 1, 0)
    asm, con, eng, idm = feature_computer(glcm_0)
    return asm, con, eng, idm

def process_directory(directory_path, output_file):
    with open(output_file, 'w') as f:
        for filename in os.listdir(directory_path):
            if filename.endswith('.png'):
                image_path = os.path.join(directory_path, filename)
                asm, con, eng, idm = process_image(image_path)
                f.write(f'{filename}: ASM={asm}, Contrast={con}, Entropy={eng}, IDM={idm}\n')

if __name__ == '__main__':
    directory_path = '/opt/data/private/wangjuntong/datasets/progan_train/car/0_real'
    output_file = 'huidu.txt'
    process_directory(directory_path, output_file)
