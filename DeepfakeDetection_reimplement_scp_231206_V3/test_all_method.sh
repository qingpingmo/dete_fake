#!/bin/bash

pwd=$(cd $(dirname $0); pwd)
echo pwd: $pwd

wget http://10.126.56.37:8080/directlink/1/CNNDetection/pytorch18_231207.tar.gz
tar -zxvf pytorch18_231207.tar.gz

mkdir datasets
cd datasets

wget http://10.126.56.37:8080/directlink/1/CNNDetection/CNN_synth_testset.zip          ## for eval_test8gan.py 
unzip CNN_synth_testset.zip -d CNN_synth_testset

wget http://10.126.56.37:8080/directlink/1/CNNDetection/dire_bedroom_imagenet.tar.gz   # for eval_test8gan_iccv23diffusion_v2.py
tar -zxvf dire_bedroom_imagenet.tar.gz


## http://10.126.56.37:8080/directlink/1/CNNDetection/ojha_diffusion_datasets.zip 这个和下面的是一个数据，但是下面的经过了整理

wget http://10.126.56.37:8080/directlink/1/CNNDetection/ojha_diffusion_datasets_231207.tar.gz    # for eval_test8gan_ojhadiffusion.py
tar -zxvf ojha_diffusion_datasets_231207.tar.gz

wget http://10.126.56.37:8080/directlink/1/CNNDetection/testDP_20231113.tar.gz  # for eval_testDPMclass_V2.py  此数据集里的ddpm有cifar 32x32
tar -zxvf testDP_20231113.tar.gz

wget http://10.126.56.37:8080/directlink/1/CNNDetection/test_onmygen_2k_20231113.tar.gz # for eval_test_mygen9GANs.py
tar -zxvf test_onmygen_2k_20231113.tar.gz 

GPU=0

# 修改各个文件夹下eval_test8gan.py, eval_test8gan_iccv23diffusion_v2.py eval_test8gan_ojhadiffusion.py 
# 内的opt.dataroot opt.model_path opt.batch_size


echo "######################################################################################"
echo "##### Cnn-generated images are surprisingly easy to spot... for now. In CVPR 2020 ####"
echo "######################################################################################"
echo "========================"
echo "==Testing CNNDetection=="
echo "========================"
cd CNNDetection

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan.py --dataroot $pwd/datasets/CNN_synth_testset
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_iccv23diffusion_v2.py --dataroot $pwd/datasets/dire_bedroom_imagenet/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_ojhadiffusion.py  --dataroot $pwd/datasets/ojha_diffusion_datasets/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_testDPMclass_V2.py  --dataroot $pwd/datasets/testDP_20231113/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test_mygen9GANs.py  --dataroot $pwd/datasets/test_onmygen_2k/
echo
cd $pwd



echo "########################################################################################################"
echo "##### Towards universal fake image detectors that generalize across generative models. In CVPR 2023 ####"
echo "########################################################################################################"
echo "==============================="
echo "==Testing UniversalFakeDetect=="
echo "==============================="
cd CNNDetection_UniversalFakeDetect

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan.py --dataroot $pwd/datasets/CNN_synth_testset
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_iccv23diffusion_v2.py  --dataroot $pwd/datasets/dire_bedroom_imagenet/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_ojhadiffusion.py --dataroot $pwd/datasets/ojha_diffusion_datasets/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_testDPMclass_V2.py  --dataroot $pwd/datasets/testDP_20231113/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test_mygen9GANs.py  --dataroot $pwd/datasets/test_onmygen_2k/
echo
cd $pwd


echo "##############################################################################################################################################"
echo "#####  Watch your up-convolution: Cnn based generative deep neural networks are failing to reproduce spectral distributions. In CVPR 2020 ####"
echo "##############################################################################################################################################"
echo "================="
echo "==Testing Durall=="
echo "================="
cd MyCNNDetection2_2_Durall


CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan.py  --dataroot $pwd/datasets/CNN_synth_testset
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_iccv23diffusion_v2.py  --dataroot $pwd/datasets/dire_bedroom_imagenet/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_ojhadiffusion.py --dataroot $pwd/datasets/ojha_diffusion_datasets/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_testDPMclass_V2.py  --dataroot $pwd/datasets/testDP_20231113/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test_mygen9GANs.py  --dataroot $pwd/datasets/test_onmygen_2k/
echo
cd $pwd



echo "#######################################################################################"
echo "#####   Leveraging frequency analysis for deepfake image recognition. In ICML 2020 ####"
echo "#######################################################################################"
echo "================="
echo "==Testing Frank=="
echo "================="
cd MyCNNDetection2_2_Frank


CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan.py --dataroot $pwd/datasets/CNN_synth_testset
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_iccv23diffusion_v2.py  --dataroot $pwd/datasets/dire_bedroom_imagenet/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_ojhadiffusion.py --dataroot $pwd/datasets/ojha_diffusion_datasets/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_testDPMclass_V2.py  --dataroot $pwd/datasets/testDP_20231113/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test_mygen9GANs.py  --dataroot $pwd/datasets/test_onmygen_2k/
echo
cd $pwd




echo "######################################################################################################"
echo "#####   What makes fake images detectable? understanding properties that generalize. In ECCV 2020 ####"
echo "######################################################################################################"
echo "==========================="
echo "==Testing patch-forensics=="
echo "==========================="
cd MyCNNDetection2_2_patch-forensics


CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan.py --dataroot $pwd/datasets/CNN_synth_testset
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_iccv23diffusion_v2.py  --dataroot $pwd/datasets/dire_bedroom_imagenet/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_ojhadiffusion.py --dataroot $pwd/datasets/ojha_diffusion_datasets/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_testDPMclass_V2.py  --dataroot $pwd/datasets/testDP_20231113/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test_mygen9GANs.py  --dataroot $pwd/datasets/test_onmygen_2k/
echo
cd $pwd



echo "########################################################################################################"
echo "#####   Thinking in frequency: Face forgery detection by mining frequency-aware clues. In ECCV 2020 ####"
echo "########################################################################################################"
echo "================="
echo "==Testing F3Net=="
echo "================="
cd MyCNNDetection2_2_usingF3Net1_V100


CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan.py --dataroot $pwd/datasets/CNN_synth_testset
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_iccv23diffusion_v2.py  --dataroot $pwd/datasets/dire_bedroom_imagenet/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test8gan_ojhadiffusion.py --dataroot $pwd/datasets/ojha_diffusion_datasets/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_testDPMclass_V2.py  --dataroot $pwd/datasets/testDP_20231113/
echo

CUDA_VISIBLE_DEVICES=$GPU $pwd/pytorch18/bin/python3 eval_test_mygen9GANs.py  --dataroot $pwd/datasets/test_onmygen_2k/
echo
cd $pwd


