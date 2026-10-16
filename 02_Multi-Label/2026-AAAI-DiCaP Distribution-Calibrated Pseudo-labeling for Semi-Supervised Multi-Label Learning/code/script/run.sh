device_id=6
dataset_dir='./data'

for lb_ratio in 0.05
do
    for dataset_name in 'voc'
    do

    CUDA_VISIBLE_DEVICES=$device_id python warm_up.py \
    --dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
    --net resnet50 --loss_lb asl --warmup_epochs 12 --lr 1e-4 --output ours 

    CUDA_VISIBLE_DEVICES=$device_id python main.py \
    --dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
    --net resnet50 --loss_lb asl --warmup_epochs 12 --lr 1e-4 \
    --output ours --method main

    CUDA_VISIBLE_DEVICES=$device_id python fine_tune.py \
    --dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
    --net resnet50 --loss_lb asl --output ours  \
    --method main --FT_method fine_tune --FT_lr 1e-4 --FT_epochs 20
    done
done