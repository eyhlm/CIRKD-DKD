CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 \
    train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --dataset voc \
    --crop-size 512 512 \
    --data /home/cwh/CIRKD/mydataset/VOCAug \
    --lambda-memory-pixel 0.01 \
	--lambda-memory-region 0.01 \
    --save-dir /home/cwh/CIRKD/checkpoint/VOCAug-cirkd \
    --log-dir /home/cwh/CIRKD/logs/deeplabv3_resnet18_VOC \
    --teacher-pretrained /home/cwh/CIRKD/pretrained_model/deeplabv3_resnet101_pascal_aug_best_model.pth \
    --student-pretrained-base /home/cwh/CIRKD/pretrained-backbone/resnet18-imagenet.pth


# CUDA_VISIBLE_DEVICES=0,1,2 \
# python -m torch.distributed.launch --nproc_per_node=3 eval.py \
#     --model deeplabv3 \
#     --backbone resnet18 \
#     --dataset voc \
#     --data /home/cwh/CIRKD/mydataset/VOCAug \
#     --save-dir /home/cwh/CIRKD/checkpoint/VOCAug-cirkd \
#     --pretrained [your pretrained model path]