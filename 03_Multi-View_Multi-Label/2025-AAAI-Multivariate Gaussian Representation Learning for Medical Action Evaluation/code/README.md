# GaussMedAct: Multivariate Gaussian based Representation Learning for Medical Action Evaluation
### Overview

This paper addresses critical dataset and methodology gaps in medical action evaluation. We introduce `CPREval-6k`, a much-needed fine-grained medical dataset that incorporates expert-validated hierarchical annotations which capture subtle error patterns. Through comprehensive comparative studies, we demonstrate that our proposed framework using spatiotemporal Gaussian mixture representation in decoupled joint and bone spaces outperforms both RGB- and skeleton- based models in accuracy while achieving significant computational cost reduction. The framework also exhibits robustness to cross-dataset generalization. These dual contributions establish a new foundation for the real-time evaluation of CPR, with potential applications that extend to other important medical and clinical assessments.

### Repository Structure

```bash
GAUSSMEDACT/
├── analysis/                 
│   ├── freq_analysis.py             # Video frequency analysis
│   ├── Main_Additional_Label.ipynb  # Analysis of main and auxiliary labels
│   └── Raw_Data.ipynb               # Detailed dataset analysis
├── dataset/                  
│   ├── raw/                         # Raw dataset (will be published later)
│   ├── train_list.json      
│   └── val_list.json         
├── scripts/                  
│   ├── ablation_fusion.py           # Different fusion method experiments
│   ├── ablation_loss.py             # Different loss function experiments
│   ├── only_mgr.py                  # Experiment using only MGR
│   └── without_mgr.py               # Experiment without MGR (direct fusion)
├── src/                      
│   ├── encoder/              
│   │   ├── HSE.py                   # Hybrid Spatial Encoding implementation
│   │   └── MGR.py                   # Multivariate Gaussian Representation implementation
│   └── models/               
│       ├── attn/             
│       │   ├── attn_cross.py        # Cross-attention implementation
│       │   └── attn_self.py         # Self-attention implementation
│       ├── model.py                 # CNN-based model implementation
│       └── utils.py          
├── tools/                    
│   ├── data/                        # Data processing utilities
│       │   ├── custom_dataset.py    # Generate dataset from skeleton prediction files
│       │   └── reFrame.py           # modify video parameter
│   ├── evaluation.py                # Model evaluation script
│   └── train.py                     # Model training script
├── LICENSE                   
└── README.md                 
```

## Usage

### Training

To train the model as described in the paper:

```bash
python tools/train.py
```

### Evaluation

To evaluate a trained model:

```bash
python tools/evaluation.py --checkpoint /path/to/model
```

## Dataset

The raw and processed dataset will be made publicly available post-publication. Currently, the repository includes training and validation splits for reference.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
