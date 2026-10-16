# [AAAI'26]DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning

This repository is the official PyTorch implementation for the **AAAI 2026** paper "DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning", its method named **DiCaP**.

## 1. Abstract

Semi-supervised multi-label learning (SSMLL) aims to address the challenge of limited labeled data availability in multi-label learning (MLL) by leveraging unlabeled data to improve the model’s performance. While pseudo-labeling has become a dominant strategy in SSMLL, most existing methods assign **equal weights** to all pseudo-labels regardless of their quality, which can amplify the impact of noisy or uncertain predictions and degrade the overall performance. In this paper, we **theoretically** verify that the optimal weight for a pseudo-label should reflect its correctness likelihood. **Empirically**, we observe that on the same dataset, the correctness likelihood distribution of unlabeled data remains stable, even as the number of labeled training samples varies. Building on this insight, we propose **Di**stribution-**Ca**librated **P**seudo-labeling (**DiCaP**), a correctness-aware framework that estimates posterior precision to calibrate pseudo-label weights. We further introduce a dual-thresholding mechanism to separate confident and ambiguous regions: confident samples are pseudo-labeled and weighted accordingly, while ambiguous ones are explored by unsupervised contrastive learning. Experiments conducted on multiple benchmark datasets verify that our method achieves consistent improvements, surpassing state-of-the-art methods by up to **4.27%**. 

## 2. Requirements
The code requires `python>=3.8` and the following packages.
```
numpy==2.3.2
pandas==1.4.2
Pillow==11.3.0
randaugment==1.0.2
scikit_learn==1.7.1
termcolor==3.1.0
timm==1.0.19
torch==2.0.1
torchvision==0.15.2
```
These packages can be installed directly by running the following command:
```
pip install -r requirements.txt
```
Note that all the experiments are conducted under one single <b>RTX 3090</b>, so the results may be a little different with the original paper when you use a different gpu.

## 3. Experiments with DiCaP

### 3.1 Get data

See the "README.md" file in the "data" directory for instructions on downloading and setting up the datasets.

### 3.2 Reproduce the results of DiCaP

In order to reproduce the results of DiCaP, you need to change the hyper-parameters in the bash scripts ([./script/run.sh](./script/run.sh)) as follows.


```bash
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
```

For example, the above provides how to run the results of DiCaP on voc with 0.05 labeled ratio, please change the value of `dataset_dir` as yours and run the following command:
```shell
bash ./script/run.sh
```

## 4. Reference
This codebase refers to D2L [[link](https://github.com/JiahaoXxX/SSMLL-D2L_MAT)], DESP [[link](https://github.com/yihanxxu/DESP)], BBAM [[link](https://github.com/changchunli/SSMLL-BBAM)], thank them!
