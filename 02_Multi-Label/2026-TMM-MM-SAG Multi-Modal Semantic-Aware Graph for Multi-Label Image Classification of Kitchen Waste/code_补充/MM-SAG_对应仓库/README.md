#  MM-SAG: Multi-Modal Semantic-Aware Graph for Multi-Label Image Classification of Kitchen Waste [PyTorch]



Hai Qin, Jintao Li, Qiaokang Liang

*School of Electrical and Information Engineering, Hunan University, Changsha, China.*



⭐ If MM-SAG is helpful to you, please star this repo. Thanks! 

## 📝 Abstract

Multi-label image classification aims to recognize multiple object labels within an image. In the field of intelligent waste sorting, efficient classification can enhance the accuracy of 
robotic sorting. However, most existing waste classification tasks are single-label, and there is limited research on multi-label classification of urban kitchen waste, especially in complex backgrounds and diverse categories. To address this, we propose a Multi-Modal Semantic-Aware Graph (MM-SAG) framework, which includes a semantic-aware module designed for instance-level label relationship mining. The captured semantic features are then processed through graph convolution to generate a label correlation matrix, enhancing the efficiency and effectiveness of label correlation mining. To improve the integration of visual and linguistic modalities, we design an improved multi-head attention mechanism module. This module re-encodes and aligns visual and textual features, further enhancing feature extraction capabilities. Experimental results show that our proposed method achieves a mean Average Precision (mAP) of 83.1\% on the MLKW dataset, delivering state-of-the-art performance. The method's strong generalization capability is also validated on public datasets VOC2007 and MS-COCO. The source code and dataset are available at our GitHub repository: https://github.com/yulle13/MMSAG.git.

##  Overview

![arch](images\fig1.png)

## ⚙ Environment

```shell
torch==2.2.0 
numpy==1.24.4
opencv-python==4.2.0
scikit-image==0.21.0
```


## 🔥 Dataset

Download the [dataset] (https://drive.google.com/file/d/1yESiv5qgWK5vXynWBYiuEq9Ose0yvphe/view?usp=drive_link) 


## ⚡Results

![comp1](images\fig4.png)

![comp2](images\fig6.png)

