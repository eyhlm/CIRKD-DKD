CUDA_VISIBLE_DEVICES=2,3 \
    python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501\
    _train_dkd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --dataset camvid \
    --crop-size 360 360 \
    --data /home/cwh/CIRKD/mydataset/CamVid/CamVid \
    --save-dir /home/cwh/CIRKD/checkpoint/CamVid-cirkd-dkd_lambda-kd=0.02 \
    --log-dir /home/cwh/CIRKD/checkpoint/CamVid-cirkd-dkd_lambda-kd=0.02 \
    --teacher-pretrained /home/cwh/CIRKD/pretrained_model/deeplabv3_resnet101_camvid_best_model.pth \
    --student-pretrained-base /home/cwh/CIRKD/pretrained-backbone/resnet18-imagenet.pth \
    --batch-size 8 \
    --lambda-kd 0.02 \
    --lr 0.02

# CUDA_VISIBLE_DEVICES=0,1 \
#   python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501\
#   eval.py \
#   --model deeplabv3 \
#   --backbone resnet18 \
#   --dataset camvid \
#   --data /home/cwh/CIRKD/mydataset/CamVid/CamVid \
#   --save-dir /home/cwh/CIRKD/logs/deeplabv3_resnet18_CamVid-cwd \
#   --pretrained /home/cwh/CIRKD/checkpoint/CamVid-cwd/kd_deeplabv3_resnet18_camvid_best_model.pth \
#   --gpu-id 0,1