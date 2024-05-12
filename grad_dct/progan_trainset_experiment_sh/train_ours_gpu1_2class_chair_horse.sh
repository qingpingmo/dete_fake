
echo "Training with model: ours_chair_horse on GPU 1"

# 在这里使用 $model1 $model2 作为 --name 参数传递给 train2.py 脚本
CUDA_VISIBLE_DEVICES=1 python ../train2.py --name "ours_chair_horse" \
--blur_prob 0.1 --blur_sig 0.0,3.0 \
--jpg_prob 0.1 --jpg_method cv2,pil \
--jpg_qual 30,100 \
--dataroot /root/rstao/datasets/ \
--dataset1_root "/root/rstao/datasets/progan_train/chair" \
--dataset2_root "/root/rstao/datasets/progan_train/horse" \
--classes chair,horse --batch_size 128 --delr_freq 10 --lr 0.0002 --niter 100 \
--train_split progan_train --val_split progan_val --mode_method ours
# 标记任务为已完成
# 等待一段时间，如果需要的话（可选）
# sleep 5