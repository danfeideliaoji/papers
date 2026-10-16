# Dual View Alignment Learning with Hierarchical Prompt for Class-Imbalance Multi-Label Image Classification[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/open-vocabulary-multi-label-classification/multi-label-zero-shot-learning-on-nus-wide)](https://paperswithcode.com/sota/multi-label-zero-shot-learning-on-nus-wide?p=open-vocabulary-multi-label-classification)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/open-vocabulary-multi-label-classification/multi-label-zero-shot-learning-on-open-images)](https://paperswithcode.com/sota/multi-label-zero-shot-learning-on-open-images?p=open-vocabulary-multi-label-classification)

This is the official PyTorch implementation for our paper **"Dual View Alignment Learning with Hierarchical Prompt for Class-Imbalance Multi-Label Image Classification" (HP-DVAL)**.


This repository provides code for both **Long-Tailed (LT)** and **Few-Shot Learning (FSL)** settings for multi-label image classification on common datasets.

![Framework](figures/mkt.jpg)

## Requirements

Please install the required packages first:

## Setup

```bash

pip install -r requirements.txt```bash

```pip install -r requirements.txt

```

## Datasets Preparation

## Preparation

Ensure you have downloaded the required datasets (e.g., COCO, NUS-WIDE, VOC) and placed them correctly in the `data/` and `idx/` directories before running the scripts.

1. Download pretrained VLP(ViT-B/16) model from [OpenAI CLIP](https://github.com/openai/CLIP).

---

2. Download images of NUS-WIDE dataset  from [NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html).

## 1. Long-Tailed (LT) Tasks

3. Download annotations following the [BiAM](https://github.com/akshitac8/BiAM) from [here](https://drive.google.com/drive/folders/1jvJ0FnO_bs3HJeYrEJu7IcuilgBipasA?usp=sharing).

For the LT task, the training is divided into two stages:

- **First Stage**: Standard prompt learning or base training.4. Download other files from [here](https://drive.google.com/drive/folders/1kTb83_p92fM04OAkGyiHypOgwtxc4wVa?usp=sharing).

- **Second Stage**: Dual view alignment and hierarchical prompt tuning.

The organization of the dataset directory is shown as follows.

### Training on COCO-LT

python coco_runner.py --dataset coco-lt --lr 1e-5 --loss_function bce --topk 32 --alpha 0.4 


**Second Stage (HP-DVAL):** 

python coco_runner_dual.py --dataset coco-lt --lr 1e-5 --loss_function bce --topk 32


To evaluate a trained checkpoint on the COCO-LT evaluation se

python coco_test.py --resume path/to/your/checkpoint.ckpt --dataset coco-lt```


## 2. Few-Shot Learning (FSL) Tasks     

python fsl_runner.py --dataset voc --lr 1e-5 --topk 32 --alpha 0.4


To evaluate the FSL performance given a trained checkpoint:  

python fsl_eval.py --dataset voc --resume path/to/your/fsl_checkpoint.ckpt --shot 5 --lr 1e-5

```## Inference on A Single Image

---python3 inference.py 

