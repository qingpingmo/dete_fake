#!/bin/bash

# models 数组定义
models=('airplane' 'bicycle' 'bird' 'boat' 'bottle' 'bus' 'car' 'cat' 'chair' 'cow' 'diningtable' 'dog' 'horse' 'motorbike' 'person' 'pottedplant' 'sheep' 'sofa' 'train' 'tvmonitor')
output_directory="./our_model_finished_txt"

# 循环遍历单个类别
for model in "${models[@]}"; do
    # 检查任务是否已经运行，如果是则跳过
    if [ -e "${output_directory}/completed_${model}_our_model.txt" ]; then
        echo "Task for $model already completed. Skipping..."
        continue
    fi
    touch "${output_directory}/completed_${model}_our_model.txt"
    # 打印当前类别（可选）
    echo "Training with model: $model on GPU 3"

    # 在这里使用 $model 作为 --name 参数传递给 train2.py 脚本
    CUDA_VISIBLE_DEVICES=3 python ../train2.py --name "ours_${model}" \
    --blur_prob 0.1 --blur_sig 0.0,3.0 \
    --jpg_prob 0.1 --jpg_method cv2,pil \
    --jpg_qual 30,100 \
    --dataroot /root/rstao/datasets/ \
    --dataset1_root "/root/rstao/datasets/progan_train/$model" \
    --dataset2_root "/root/rstao/datasets/progan_train/$model" \
    --classes airplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor --batch_size 16 --delr_freq 10 --lr 0.002 --niter 100\
    --train_split progan_train --val_split progan_val --mode_method ours
    # 标记任务为已完成

    # 等待一段时间，如果需要的话（可选）
    sleep 5
done