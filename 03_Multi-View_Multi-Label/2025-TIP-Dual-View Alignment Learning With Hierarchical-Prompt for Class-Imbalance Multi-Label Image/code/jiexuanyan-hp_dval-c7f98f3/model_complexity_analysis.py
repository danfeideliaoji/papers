#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import clip
from thop import profile
from thop import clever_format
import numpy as np

# Import models
from models.clip_vit import CLIPVIT
from models.prompt_model import PromptLearner, CLIPVIT as PromptCLIPVIT
from models.clip_vit_dual import CLIPVIT as DualCLIPVIT

def convert_models_to_fp32(model):
    """Convert model to fp32 for compatibility"""
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    """Convert model to fp16 for CLIP compatibility"""
    for p in model.parameters():
        p.data = p.data.half()
        if p.grad:
            p.grad.data = p.grad.data.half()

def count_parameters(model):
    """Count trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def measure_inference_speed(model, input_tensor, num_runs=100, warmup_runs=10):
    """Measure inference speed"""
    model.eval()
    
    # Ensure input tensor has correct dtype
    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.float()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    return avg_time * 1000, fps  # Return time in ms and FPS

def analyze_first_stage_model(args):
    """Analyze first stage model (CLIPVIT)"""
    print("=" * 60)
    print("FIRST STAGE MODEL ANALYSIS (CLIPVIT)")
    print("=" * 60)
    
    # Load CLIP model
    clip_model, _ = clip.load(args.clip_path, jit=False)
    
    # Create dummy classnames for testing
    classnames = [f"class_{i}" for i in range(100)]
    
    # Create model args dict
    model_args = {
        'topk': args.topk,
        'alpha': args.alpha
    }
    
    # Create model
    model = CLIPVIT(model_args, classnames, clip_model)
    # Keep model in fp32 for compatibility
    convert_models_to_fp32(model)
    model = model.to(args.device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32).to(args.device)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total trainable parameters: {total_params/1e6:.2f}M")
    
    # Measure FLOPs
    flops, params = profile(model, (dummy_input,), verbose=False)
    print(f"FLOPs: {flops:,}")
    print(f"FLOPs: {flops/1e9:.2f}G")
    
    # Measure inference speed
    avg_time, fps = measure_inference_speed(model, dummy_input)
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Inference speed: {fps:.2f} FPS")
    
    return model, total_params, flops, avg_time, fps

def analyze_second_stage_model(args):
    """Analyze second stage model (PromptLearner)"""
    print("=" * 60)
    print("SECOND STAGE MODEL ANALYSIS (PromptLearner)")
    print("=" * 60)
    
    # Load CLIP model
    clip_model, _ = clip.load(args.clip_path, jit=False)
    
    # Create dummy classnames for testing
    classnames = [f"class_{i}" for i in range(100)]
    
    # Create model args dict
    model_args = {
        'n_ctx': args.n_ctx,
        'ctx_init': args.ctx_init,
        'class_token_position': args.class_token_position
    }
    
    # Create model
    model = PromptLearner(model_args, classnames, clip_model)
    # Keep model in fp32 for compatibility
    convert_models_to_fp32(model)
    model = model.to(args.device)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total trainable parameters: {total_params/1e6:.2f}M")
    
    # For prompt learning, we need to measure the forward pass differently
    # Create dummy input for text encoder
    dummy_input = torch.randint(0, 1000, (1, 77)).to(args.device)  # Tokenized text
    
    # Measure FLOPs (approximate for text encoder)
    try:
        flops, params = profile(model, (dummy_input,), verbose=False)
        print(f"FLOPs: {flops:,}")
        print(f"FLOPs: {flops/1e9:.2f}G")
    except:
        print("FLOPs measurement not available for this model")
        flops = 0
    
    # Measure inference speed
    avg_time, fps = measure_inference_speed(model, dummy_input)
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Inference speed: {fps:.2f} FPS")
    
    return model, total_params, flops, avg_time, fps

def analyze_dual_model(args):
    """Analyze dual model (CLIPVIT from clip_vit_dual.py)"""
    print("=" * 60)
    print("DUAL MODEL ANALYSIS (CLIPVIT Dual)")
    print("=" * 60)
    
    # Load CLIP model
    clip_model, _ = clip.load(args.clip_path, jit=False)
    
    # Create dummy classnames for testing
    classnames = [f"class_{i}" for i in range(100)]
    
    # Create model args dict
    model_args = {
        'topk': args.topk,
        'alpha': args.alpha
    }
    
    # Create model
    model = DualCLIPVIT(model_args, classnames, clip_model)
    # Keep model in fp32 for compatibility
    convert_models_to_fp32(model)
    model = model.to(args.device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32).to(args.device)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total trainable parameters: {total_params/1e6:.2f}M")
    
    # Measure FLOPs
    flops, params = profile(model, (dummy_input,), verbose=False)
    print(f"FLOPs: {flops:,}")
    print(f"FLOPs: {flops/1e9:.2f}G")
    
    # Measure inference speed
    avg_time, fps = measure_inference_speed(model, dummy_input)
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Inference speed: {fps:.2f} FPS")
    
    return model, total_params, flops, avg_time, fps

def analyze_prompt_clip_model(args):
    """Analyze prompt CLIP model (CLIPVIT from prompt_model.py)"""
    print("=" * 60)
    print("PROMPT CLIP MODEL ANALYSIS (CLIPVIT from prompt_model.py)")
    print("=" * 60)
    
    # Load CLIP model
    clip_model, _ = clip.load(args.clip_path, jit=False)
    
    # Create dummy classnames for testing
    classnames = [f"class_{i}" for i in range(100)]
    
    # Create model args dict
    model_args = {
        'topk': args.topk,
        'alpha': args.alpha,
        'n_ctx': args.n_ctx,
        'ctx_init': args.ctx_init,
        'class_token_position': args.class_token_position
    }
    
    # Create model
    model = PromptCLIPVIT(model_args, classnames, clip_model)
    # Keep model in fp32 for compatibility
    convert_models_to_fp32(model)
    model = model.to(args.device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32).to(args.device)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total trainable parameters: {total_params/1e6:.2f}M")
    
    # Measure FLOPs
    flops, params = profile(model, (dummy_input,), verbose=False)
    print(f"FLOPs: {flops:,}")
    print(f"FLOPs: {flops/1e9:.2f}G")
    
    # Measure inference speed
    avg_time, fps = measure_inference_speed(model, dummy_input)
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Inference speed: {fps:.2f} FPS")
    
    return model, total_params, flops, avg_time, fps

def main(args):
    """Main analysis function"""
    print("Model Complexity Analysis for MKT-LT Project")
    print("=" * 60)
    
    # Set device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    
    # Enable cudnn benchmark for speed
    cudnn.benchmark = True
    
    results = {}
    
    # Analyze different models
    try:
        # First stage model
        results['first_stage'] = analyze_first_stage_model(args)
        
        # Second stage model
        results['second_stage'] = analyze_second_stage_model(args)
        
        # Dual model
        results['dual'] = analyze_dual_model(args)
        
        # Prompt CLIP model
        results['prompt_clip'] = analyze_prompt_clip_model(args)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model_name, (model, params, flops, time_ms, fps) in results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print(f"  Parameters: {params/1e6:.2f}M")
        print(f"  FLOPs: {flops/1e9:.2f}G")
        print(f"  Inference Time: {time_ms:.2f} ms")
        print(f"  FPS: {fps:.2f}")
    
    # Save results to file
    save_results(results, args.output_file)

def save_results(results, output_file):
    """Save results to a text file"""
    with open(output_file, 'w') as f:
        f.write("Model Complexity Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, (model, params, flops, time_ms, fps) in results.items():
            f.write(f"{model_name.upper().replace('_', ' ')}:\n")
            f.write(f"  Parameters: {params:,} ({params/1e6:.2f}M)\n")
            f.write(f"  FLOPs: {flops:,} ({flops/1e9:.2f}G)\n")
            f.write(f"  Inference Time: {time_ms:.2f} ms\n")
            f.write(f"  FPS: {fps:.2f}\n\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Complexity Analysis")
    
    # Model paths
    parser.add_argument("--clip-path", type=str, default="/data2/yanjiexuan/huggingface/openai/pretrained/ViT-B-16.pt",
                       help="Path to CLIP model")
    
    # Analysis options
    parser.add_argument("--output-file", type=str, default="model_complexity_results.txt",
                       help="Output file to save results")
    
    # Model configuration
    parser.add_argument("--topk", type=int, default=16,
                       help="Top-k value for model")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Alpha value for model")
    parser.add_argument("--n-ctx", type=int, default=4,
                       help="Number of context tokens")
    parser.add_argument("--ctx-init", type=str, default="",
                       help="Context initialization")
    parser.add_argument("--class-token-position", type=str, default="end",
                       help="Class token position")
    
    args = parser.parse_args()
    
    main(args) 