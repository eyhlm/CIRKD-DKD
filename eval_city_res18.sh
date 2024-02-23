CUDA_VISIBLE_DEVICES=0,1,2 \
  python -m torch.distributed.launch --nproc_per_node=3 eval.py \
  --model deeplabv3 \
  --backbone resnet18 \
  --data /home/cwh/CIRKD/mydataset/OpenDataLab___CityScapes/raw/ \
  --save-dir /home/cwh/CIRKD/logs \
  --pretrained /home/cwh/CIRKD/checkpoint/city-cirkd/kd_deeplabv3_resnet18_citys_best_model.pth \
  --gpu-id 0,1,2