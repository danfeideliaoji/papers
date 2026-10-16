import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ..src.utils import *
from ..src.models.model import GaussianCNN

class SkeletonDataWrapper(Dataset):
    def __init__(self, original_dataset, mode='interleaved', bone_weight=0.5):
        self.original_dataset = original_dataset
        self.mode = mode
        self.bone_weight = bone_weight
        
        print(f"Dataset mode: {mode}")
        
        if len(original_dataset) > 0:
            first_item = original_dataset[0][0]
            total_features = first_item.shape[0]
            num_frames = first_item.shape[1]
            num_channels = first_item.shape[2]
            
            print(f"Original data shape: {first_item.shape}")
            
            self.joint_indices = [i for i in range(0, total_features, 2)]
            self.bone_indices = [i for i in range(1, total_features, 2)]
            
            if len(self.joint_indices) > len(self.bone_indices) and total_features % 2 != 0:
                print(f"Joint count ({len(self.joint_indices)}) is 1 more than Bone count ({len(self.bone_indices)})")
            
            print(f"Joint features: {len(self.joint_indices)}, Bone features: {len(self.bone_indices)}")
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        data, label = self.original_dataset[idx]
        
        if self.mode == 'interleaved':
            return data, label
        
        joint_data = data[self.joint_indices]
        bone_data = data[self.bone_indices]
        
        if self.mode == 'joint_only':
            return joint_data, label
            
        elif self.mode == 'bone_only':
            return bone_data, label
            
        elif self.mode == 'concat':
            return torch.cat([joint_data, bone_data], dim=0), label
            
        elif self.mode == 'add':
            min_features = min(joint_data.shape[0], bone_data.shape[0])
            return joint_data[:min_features] + bone_data[:min_features], label
            
        elif self.mode == 'weighted_add':
            min_features = min(joint_data.shape[0], bone_data.shape[0])
            return (1 - self.bone_weight) * joint_data[:min_features] + self.bone_weight * bone_data[:min_features], label
            
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


def train_model(model, train_loader, val_loader, device, num_epochs=100, mixup_alpha=0.2, weight_decay=1e-4, scheduler_patience=8, name='', lr=0.005):
    print("Starting training process...")
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=scheduler_patience, 
        verbose=True, threshold=0.0005, min_lr=1e-6
    )

    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha, device=device)
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (lam * predicted.eq(labels_a).float() + (1 - lam) * predicted.eq(labels_b).float()).sum().item()
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f'Epoch: [{epoch+1}/{num_epochs}][{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} Acc: {100.*train_correct/train_total:.2f}% '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        epoch_loss = running_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(epoch_loss)
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, f'./checkpoint/best_cnn_model_{name}.pth')
            patience_counter = 0
            print(f'Validation accuracy improved to {best_val_acc:.4f}, model saved.')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
            
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - '
              f'Train loss: {epoch_loss:.4f}, Train acc: {train_acc:.4f}, '
              f'Val loss: {val_loss/val_total:.4f}, '
              f'Val acc: {val_acc:.4f}, Best val acc: {best_val_acc:.4f}')
    
    checkpoint = torch.load(f'./checkpoint/best_cnn_model_{name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}, validation accuracy: {checkpoint['best_val_acc']:.4f}")
    
    return model, train_losses, val_accs


def evaluate_model(model, test_loader, device, class_names=None):
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
    print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")
    print(f"Average batch inference time: {avg_inference_time*1000:.2f} ms")
    print(f"Average per-sample inference time: {inference_per_sample*1000:.2f} ms")
    print("\nPer-class performance:")
    
    for i in range(22):
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"{class_name}: Top-1: {class_accuracy_top1[i]:.4f}, Top-5: {class_accuracy_top5[i]:.4f} ({class_total[i]} samples)")
    
    best_class_top1 = max(class_accuracy_top1, key=class_accuracy_top1.get)
    worst_class_top1 = min(class_accuracy_top1, key=class_accuracy_top1.get)
    
    print(f"\nBest performing class (Top-1): {class_names[best_class_top1] if class_names else best_class_top1} "
          f"accuracy {class_accuracy_top1[best_class_top1]:.4f}")
    print(f"Worst performing class (Top-1): {class_names[worst_class_top1] if class_names else worst_class_top1} "
          f"accuracy {class_accuracy_top1[worst_class_top1]:.4f}")
    
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
    plt.savefig(f'./graph/cm/confusion_matrix_{top1_accuracy}_{top5_accuracy}.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'./graph/others/f1_scores_{top1_accuracy}_{top5_accuracy}.png', dpi=300)
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
    plt.savefig(f'./graph/others/top1_vs_top5_accuracy_{top1_accuracy}_{top5_accuracy}.png', dpi=300)
    plt.close()
    
    return top1_accuracy, top5_accuracy, cm


def custom_collate_fn(batch):
    features = torch.stack([item[0].detach() for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return features, labels


def main():
    train_dataset_raw = torch.load('./data/train_data_mgr_only.pth')
    val_dataset_raw = torch.load('./data/train_data_mgr_only.pth')
    
    modes = ['joint_only', 'bone_only', 'interleaved', 'concat', 'add', 'weighted_add']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 64
    num_classes = 22
    
    class_names = get_class_names()
    
    selected_mode = 'joint_only'
    
    print(f"\n==== Processing mode: {selected_mode} ====")
    
    if selected_mode == 'weighted_add':
        train_dataset = SkeletonDataWrapper(train_dataset_raw, mode=selected_mode, bone_weight=0.6)
        val_dataset = SkeletonDataWrapper(val_dataset_raw, mode=selected_mode, bone_weight=0.6)
    else:
        train_dataset = SkeletonDataWrapper(train_dataset_raw, mode=selected_mode)
        val_dataset = SkeletonDataWrapper(val_dataset_raw, mode=selected_mode)
    
    sample_data = train_dataset[0][0]
    print(f"Processed data shape: {sample_data.shape}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    model = GaussianCNN(
        num_features=sample_data.shape[0],
        num_frames=sample_data.shape[1],
        channels=sample_data.shape[2],
        num_classes=num_classes
    )
    model.to(device)
    
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    model, train_losses, val_accs = train_model(
        model, train_loader, val_loader, device, 
        num_epochs=100, 
        weight_decay=0.0003, 
        name=f'cnn_{selected_mode}'
    )
    
    top1_accuracy, top5_accuracy, cm = evaluate_model(model, val_loader, device, class_names)
    
    print(f"Processing mode {selected_mode} final results: Top-1 Acc={top1_accuracy:.4f}, Top-5 Acc={top5_accuracy:.4f}")


if __name__ == "__main__":
    main()