python train2.py --name resnet_noaug6 --dataroot ./dataset/ --classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse --batch_size 64






python train2.py --name 1class-resnet_horse --dataroot ./dataset/ --classes horse --batch_size 16

python train2.py --name 2class-resnet_horse_chair --dataroot ./dataset/ --classes horse,chair --batch_size 16


python train2.py --name 4class-resnet_horse_chair_car_cat --dataroot ./dataset/ --classes car,cat,chair,horse --batch_size 64


python train2.py --name 4class-resnet_cow_diningtable_person_horse --dataroot ./dataset/ --classes cow,diningtable,person,horse --batch_size 64



python train2.py --name 4class-resnet_bicycle_diningtable_person_horse --dataroot ./dataset/ --classes bicycle,diningtable,person,horse --batch_size 64


python train2.py --name 4class-resnet_bicycle_diningtable_person_horse --dataroot ./dataset/ --classes bicycle,sofa,person,horse --batch_size 64

CUDA_VISIBLE_DEVICES=0 python train3.py --name 4class-resnet_car_cat_chair_horse --dataroot ./dataset/ --classes car,cat,chair,horse --batch_size 64 --delr_freq 7 --lr 0.0001 


CUDA_VISIBLE_DEVICES=1  python train2.py --name 4class-resnet_bicycle_bus_person_horse --dataroot ./dataset/ --classes bicycle,bus,person,horse --batch_size 64





CUDA_VISIBLE_DEVICES=0 python train4.py --name 20class-resnet --dataroot ./dataset/ --classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse --batch_size 64 --delr_freq 20 --lr 0.0001 


CUDA_VISIBLE_DEVICES=0 python train4.py --name 1class-resnet-horse --dataroot ./dataset/ --classes horse --batch_size 64 --delr_freq 10 --lr 0.0001 


CUDA_VISIBLE_DEVICES=0 python train4.py --name 4class-resnet_car_cat_chair_horse --dataroot ./dataset/ --classes car,cat,chair,horse --batch_size 64 --delr_freq 10 --lr 0.0001 


CUDA_VISIBLE_DEVICES=0 python train4.py --name 2class-resnet_chair_horse --dataroot ./dataset/ --classes chair,horse --batch_size 64 --delr_freq 10 --lr 0.0001 

# python train2.py --name resnet_noaug4 --dataroot ./dataset/ --classes train,sofa,horse
95.5	99.4	80.6	90.6	77.4	93.0	63.5	60.5	59.4	59.9	99.6	100.	53.0	49.1	70.4	81.5

1	2	3	4	5


CUDA_VISIBLE_DEVICES=4 python train4.py --name 4class-resnet_car_cat_chair_horse_bs32_ --dataroot /opt/data/private/tcc/data/data/CNNDetection/ --classes car,cat,chair,horse --batch_size 32 --delr_freq 4 --lr 0.0002 --niter 40 --lnum 64 --delr 0.9 --pth random_sobel --seed 111