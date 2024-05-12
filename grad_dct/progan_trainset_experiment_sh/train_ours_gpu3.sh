#!/bin/bash

# models 数组定义
# models=('biggan' 'crn' 'cyclegan' 'deepfake' 'gaugan' 'imle' 'progan' 'san' 'seeingdark' 'stargan' 'stylegan' 'stylegan2' 'whichfaceisreal')
# models=('progan' 'stylegan' 'stylegan2' 'cyclegan' 'stargan' 'gaugan' 'deepfake' 'biggan')
models=('airplane' 'bicycle' 'bird' 'boat' 'bottle' 'bus' 'car' 'cat' 'chair' 'cow' 'diningtable' 'dog' 'horse' 'motorbike' 'person' 'pottedplant' 'sheep' 'sofa' 'train' 'tvmonitor')
output_directory="./progan_trainset_our_finished_txt"
# 循环遍历两个元素的所有组合
for ((i=0; i<${#models[@]}; i++)); do
    for ((j=i+1; j<${#models[@]}; j++)); do
        model1=${models[i]}
        model2=${models[j]}

        # 检查任务是否已经运行，如果是则跳过
        if [ -e "${output_directory}/completed_${model1}_${model2}_ours.txt" ]; then
            echo "Task for $model1 and $model2 already completed. Skipping..."
            continue
        fi
        touch "${output_directory}/completed_${model1}_${model2}_ours.txt"
        # 打印当前组合（可选）
        echo "Training with model: $model1 and $model2 on GPU 0"

        # 在这里使用 $model1 $model2 作为 --name 参数传递给 train2.py 脚本
        CUDA_VISIBLE_DEVICES=3 python ../train2.py --name "ours_${model1}_${model2}" \
        --blur_prob 0.1 --blur_sig 0.0,3.0 \
        --jpg_prob 0.1 --jpg_method cv2,pil \
        --jpg_qual 30,100 \
        --dataroot /opt/data/private/rstao/code/DeepfakeDetection/datasets/CNNDetection/ \
        --dataset1_root "/opt/data/private/rstao/code/DeepfakeDetection/datasets/CNNDetection/progan_train/$model1" \
        --dataset2_root "/opt/data/private/rstao/code/DeepfakeDetection/datasets/CNNDetection/progan_train/$model2" \
        --classes chair,horse --batch_size 128 --delr_freq 10 --lr 0.0002 --niter 100\
        --train_split progan_train --val_split progan_val --mode_method ours
        # 标记任务为已完成
        # 等待一段时间，如果需要的话（可选）
        # sleep 5
    done
done