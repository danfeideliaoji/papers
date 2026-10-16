import time
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import *
from ..src.models.model import GaussianCNN
from torch.utils.data import DataLoader, Dataset
import os


def evaluate_model(model, test_loader, device, class_names=None, name='default'):
    model.eval()
    test_preds = []
    test_true = []
    
    inference_times = []
    
    class_correct_top1 = [0] * 22
    class_correct_top5 = [0] * 22
    class_total = [0] * 22
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            values, indices = torch.sort(outputs, dim=1, descending=True)
            top1_preds = indices[:, 0]
            
            k = min(5, outputs.size(1))
            top5_preds = indices[:, :k]
            
            correct_top1 = top1_preds.eq(labels)
            top1_correct += correct_top1.sum().item()
            
            correct_top5 = torch.zeros_like(labels, dtype=torch.bool)
            for i in range(labels.size(0)):
                if labels[i] in top5_preds[i]:
                    correct_top5[i] = True
            
            top5_correct += correct_top5.sum().item()
            total += labels.size(0)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if correct_top1[i]:
                    class_correct_top1[label] += 1
                if correct_top5[i]:
                    class_correct_top5[label] += 1
            
            test_preds.extend(top1_preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    
    cm = confusion_matrix(test_true, test_preds)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    inference_per_sample = avg_inference_time / test_loader.batch_size
    
    class_accuracy_top1 = {i: (class_correct_top1[i] / class_total[i] if class_total[i] > 0 else 0) for i in range(22)}
    class_accuracy_top5 = {i: (class_correct_top5[i] / class_total[i] if class_total[i] > 0 else 0) for i in range(22)}
    
    print("\n===== Model Evaluation Report =====")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Average batch inference time: {avg_inference_time*1000:.2f} ms")
    print(f"Average inference time per sample: {inference_per_sample*1000:.2f} ms")
    print("\nPerformance by class:")
    
    for i in range(22):
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"{class_name}: Top-1: {class_accuracy_top1[i]:.4f}, Top-5: {class_accuracy_top5[i]:.4f} ({class_total[i]} samples)")
    
    best_class_top1 = max(class_accuracy_top1, key=class_accuracy_top1.get)
    worst_class_top1 = min(class_accuracy_top1, key=class_accuracy_top1.get)
    
    print(f"\nBest performing class (Top-1): {class_names[best_class_top1] if class_names else best_class_top1} "
          f"with accuracy {class_accuracy_top1[best_class_top1]:.4f}")
    print(f"Worst performing class (Top-1): {class_names[worst_class_top1] if class_names else worst_class_top1} "
          f"with accuracy {class_accuracy_top1[worst_class_top1]:.4f}")
    
    plt.figure(figsize=(20, 16))
    sns.set(font_scale=1.2)
    cm_display = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=class_names if class_names else None,
                            yticklabels=class_names if class_names else None)
    plt.xlabel('Predicted Class', fontsize=15)
    plt.ylabel('True Class', fontsize=15)
    plt.title(f'Confusion Matrix (Top-1 Acc: {top1_accuracy:.4f}, Top-5 Acc: {top5_accuracy:.4f})', fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'./graph/cm/confusion_matrix_{name}_{top1_accuracy:.4f}_{top5_accuracy:.4f}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    report = classification_report(test_true, test_preds, target_names=class_names if class_names else None, output_dict=True)
    
    f1_scores = [report[str(i)]['f1-score'] for i in range(22)] if class_names is None else [report[name]['f1-score'] for name in class_names]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(22), f1_scores)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores by Class')
    plt.xticks(range(22), class_names if class_names else range(22), rotation=90)
    plt.tight_layout()
    plt.savefig(f'./graph/others/f1_scores_{name}_{top1_accuracy:.4f}_{top5_accuracy:.4f}.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    categories = list(range(22))
    
    top1_values = [class_accuracy_top1[i] for i in categories]
    top5_values = [class_accuracy_top5[i] for i in categories]
    
    bar_width = 0.35
    index = np.arange(len(categories))
    
    plt.bar(index, top1_values, bar_width, label='Top-1 Accuracy')
    plt.bar(index + bar_width, top5_values, bar_width, label='Top-5 Accuracy')
    
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Top-1 vs Top-5 Accuracy Comparison')
    plt.xticks(index + bar_width/2, class_names if class_names else categories, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./graph/others/top1_vs_top5_accuracy_{name}_{top1_accuracy:.4f}_{top5_accuracy:.4f}.png', dpi=300)
    plt.close()
    
    return top1_accuracy, top5_accuracy, cm


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate GaussianCNN model')
    parser.add_argument('--data_path', type=str, default='data/val_data_bone_joint_cross_full.pt',
                        help='Path to validation dataset (default: data/val_data_bone_joint_cross_full.pt)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/best.pth',
                        help='Path to model checkpoint (default: checkpoint/best.pth)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--num_joints', type=int, default=117,
                        help='Number of joints in the model (default: 117)')
    parser.add_argument('--gmm_components', type=int, default=22,
                        help='Number of GMM components (default: 22)')
    parser.add_argument('--gaussian_params', type=int, default=10,
                        help='Number of Gaussian parameters (default: 10)')
    parser.add_argument('--num_classes', type=int, default=22,
                        help='Number of classes (default: 22)')
    parser.add_argument('--name', type=str, default='joint_only',
                        help='Name for saving results (default: joint_only)')
    parser.add_argument('--output_dir', type=str, default='./graph',
                        help='Directory for saving output graphs (default: ./graph)')
    
    args = parser.parse_args()

    print(f"Loading validation dataset from {args.data_path}...")
    val_dataset = torch.load(args.data_path)
    
    print("Extracting joint data...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GaussianCNN(num_joints=args.num_joints, 
                        gmm_components=args.gmm_components, 
                        gaussian_params=args.gaussian_params, 
                        num_classes=args.num_classes)
    model.to(device)
    
    print(f"Loading model checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(model)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    class_names = get_class_names()
    
    print("Evaluating model...")
    top1_accuracy, top5_accuracy, cm = evaluate_model(model, val_loader, device, class_names, name=args.name)
    print(f"Final results - Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}")


if __name__ == "__main__":
    # Create output directories
    output_dirs = ['./checkpoint', './graph/cm', './graph/others']
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    main()