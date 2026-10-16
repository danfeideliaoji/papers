# Towards-Calibrated-Multi-label-Deep-Neural-Networks

This is the code for the paper: \
[Towards Calibrated Multi-label Deep Neural Networks](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.html). \
Jiacheng Cheng and Nuno Vasconcelos. In CVPR 2024.

If you find this repo useful for your research, please cite this paper as: 
```
@InProceedings{Cheng_2024_CVPR,
    author    = {Cheng, Jiacheng and Vasconcelos, Nuno},
    title     = {Towards Calibrated Multi-label Deep Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {27589-27599}
}
```

## Dependencies
We implement our methods in PyTorch on NVIDIA GPUs. The environment requirements are as bellow:
- PyTorch, version >= 1.11.0
- torchvision, version >= 0.12.0
- timm, version >= 1.0.8

## Usage
To train DNNs with the proposed Strictly Proper Asymetric (SPA) loss, run 
```
python train.py --model-name vit_base_patch32_224 --lr 5e-5 --k0 3.0 --b0 1.0 --zeta0 5.0 
python train.py --model-name ecaresnet50t --lr 1e-3 --k0 3.0 --b0 1.0 --zeta0 5.0 
python train.py --model-name ecaresnet50t --lr 1e-3 --k0 1.0 --b0 0.0 --zeta0 1.0 
```



## Acknowledgments
Parts of implementations are based on the following public code: 
- [ASL](https://github.com/Alibaba-MIIL/ASL)
- [TwoWayMultiLabelLoss](https://github.com/tk1980/TwoWayMultiLabelLoss)



