CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --data /home/cwh/CIRKD/mydataset/OpenDataLab___CityScapes/raw/ \
    --save-dir /home/cwh/CIRKD/checkpoint/city-cirkd \
    --log-dir /home/cwh/CIRKD/logs \
    --teacher-pretrained /home/cwh/CIRKD/pretrained_model/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base /home/cwh/CIRKD/pretrained-backbone/resnet18-imagenet.pth