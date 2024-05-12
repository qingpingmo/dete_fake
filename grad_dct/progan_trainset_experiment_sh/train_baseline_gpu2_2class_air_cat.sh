CUDA_VISIBLE_DEVICES=2 python ../train2.py --name "baseline_airplane_cat" \
--blur_prob 0.1 --blur_sig 0.0,3.0 \
--jpg_prob 0.1 --jpg_method cv2,pil \
--jpg_qual 30,100 \
--dataroot /opt/data/private/rstao/code/DeepfakeDetection/datasets/CNNDetection/ \
--dataset1_root "/opt/data/private/rstao/code/DeepfakeDetection/datasets/CNNDetection/progan_train/airplane" \
--dataset2_root "/opt/data/private/rstao/code/DeepfakeDetection/datasets/CNNDetection/progan_train/cat" \
--classes chair,horse --batch_size 128 --delr_freq 10 --lr 0.0002 --niter 100\
--train_split progan_train --val_split progan_val --mode_method baseline