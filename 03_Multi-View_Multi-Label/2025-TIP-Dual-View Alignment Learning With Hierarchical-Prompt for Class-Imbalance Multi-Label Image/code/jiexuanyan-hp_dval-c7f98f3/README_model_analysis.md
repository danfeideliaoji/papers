# Model Complexity Analysis for MKT-LT Project

This directory contains scripts to analyze the computational complexity of models in the MKT-LT (Multi-Label Knowledge Transfer with Long-Tail) project.

## Scripts

### 1. `simple_model_analysis.py` (Recommended)
A simplified script that analyzes the main models used in the project:
- **CLIPVIT** (First Stage): Visual encoder based on CLIP ViT
- **PromptLearner** (Second Stage): Text prompt learning model

### 2. `model_complexity_analysis.py` (Comprehensive)
A comprehensive script that analyzes all models including:
- CLIPVIT from `clip_vit.py`
- PromptLearner from `prompt_model.py`
- CLIPVIT from `clip_vit_dual.py`
- CLIPVIT from `prompt_model.py`

## Usage

### Basic Usage
```bash
python simple_model_analysis.py
```

### With Custom Parameters
```bash
python simple_model_analysis.py \
    --clip-path /path/to/ViT-B-16.pt \
    --output-file results.txt \
    --topk 16 \
    --alpha 0.5 \
    --n-ctx 4
```

### Parameters

- `--clip-path`: Path to CLIP model file (default: "ViT-B-16.pt")
- `--output-file`: Output file to save results (default: "model_complexity_results.txt")
- `--topk`: Top-k value for model (default: 16)
- `--alpha`: Alpha value for model (default: 0.5)
- `--n-ctx`: Number of context tokens (default: 4)
- `--ctx-init`: Context initialization (default: "")
- `--class-token-position`: Class token position (default: "end")

## Output

The script will output:

1. **Parameter Count**: Total trainable parameters in millions
2. **FLOPs**: Floating point operations in billions
3. **Inference Time**: Average inference time in milliseconds
4. **FPS**: Frames per second (inference speed)

### Example Output
```
============================================================
CLIPVIT MODEL ANALYSIS (First Stage)
============================================================
Total trainable parameters: 87,000,000
Total trainable parameters: 87.00M
FLOPs: 11,200,000,000
FLOPs: 11.20G
Average inference time: 45.23 ms
Inference speed: 22.11 FPS

============================================================
PROMPT LEARNER MODEL ANALYSIS (Second Stage)
============================================================
Total trainable parameters: 2,048,000
Total trainable parameters: 2.05M
FLOPs: 1,500,000,000
FLOPs: 1.50G
Average inference time: 12.34 ms
Inference speed: 81.04 FPS
```

## Requirements

Make sure you have the following dependencies installed:
```bash
pip install torch torchvision
pip install thop
pip install clip
```

## Notes

1. **GPU Required**: The script will use GPU if available, otherwise CPU
2. **CLIP Model**: You need to download the CLIP model file (ViT-B-16.pt) or specify the correct path
3. **Memory**: Large models may require significant GPU memory
4. **Accuracy**: FLOPs measurement may not be available for all models due to complex architectures

## Troubleshooting

1. **Import Errors**: Make sure you're running the script from the project root directory
2. **CUDA Errors**: If you encounter CUDA errors, try running with CPU: `CUDA_VISIBLE_DEVICES="" python simple_model_analysis.py`
3. **Memory Errors**: Reduce batch size or use smaller input sizes if you encounter memory issues

## Model Architecture Summary

### First Stage (CLIPVIT)
- **Purpose**: Visual feature extraction using CLIP ViT backbone
- **Input**: Images (3×224×224)
- **Output**: Visual features for classification
- **Parameters**: ~87M (mostly from CLIP backbone)

### Second Stage (PromptLearner)
- **Purpose**: Learnable text prompts for better text-image alignment
- **Input**: Text tokens
- **Output**: Learned text representations
- **Parameters**: ~2M (learnable prompt tokens)

## Citation

If you use this analysis in your research, please cite the original MKT-LT paper. 