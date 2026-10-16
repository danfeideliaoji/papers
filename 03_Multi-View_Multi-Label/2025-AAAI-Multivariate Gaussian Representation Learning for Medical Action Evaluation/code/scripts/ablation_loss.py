import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from tqdm import tqdm
from ..src.utils import *
from ..src.models.model import GaussianCNN
from ..tools.train import train_model
from ..tools.evaluation import evaluate_model

os.makedirs('./graph/cm', exist_ok=True)
os.makedirs('./graph/loss', exist_ok=True)
os.makedirs('./graph/acc', exist_ok=True)
os.makedirs('./graph/others', exist_ok=True)
os.makedirs('./checkpoint', exist_ok=True)
os.makedirs('./results', exist_ok=True)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class FocalLoss(nn.Module):
    def __init__(self, classes=22, gamma=2.0, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.cls = classes
        self.size_average = size_average

    def forward(self, pred, target):
        target_one_hot = torch.zeros(pred.size(), device=pred.device)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        probs = F.softmax(pred, dim=1)
        pt = (target_one_hot * probs).sum(1)
        
        focal_weight = (1 - pt).pow(self.gamma)
        
        if self.alpha > 0:
            alpha_t = torch.ones_like(target) * self.alpha
            alpha_t = torch.where(target > 0, alpha_t, 1-self.alpha)
            focal_weight = alpha_t * focal_weight
        
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        loss = focal_weight * ce_loss
        
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()




def custom_collate_fn(batch):
    features = torch.stack([item[0].detach() for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return features, labels


def train_model(model, train_loader, val_loader, device, num_epochs=100, 
                loss_type='ce', smoothing=0.1, mixup_alpha=0.2, 
                weight_decay=1e-4, scheduler_patience=8, name='', lr=0.005):
    print(f"Starting training process... Loss function: {loss_type}")
    
    if loss_type == 'label_smoothing':
        criterion = LabelSmoothingLoss(classes=22, smoothing=smoothing)
    elif loss_type == 'focal':
        criterion = FocalLoss(classes=22, gamma=2.0, alpha=0.25)
    elif loss_type == 'multi_margin':
        criterion = nn.MultiMarginLoss()
    else:  # 'ce'
        criterion = nn.CrossEntropyLoss()
    
    use_mixup = (loss_type == 'mixup')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=scheduler_patience, 
        verbose=True, threshold=0.0005, min_lr=1e-6
    )

    best_val_acc = 0.0
    patience = 200
    patience_counter = 0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if use_mixup:
                mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha, device=device)
                outputs = model(mixed_inputs)
                loss = mixup_criterion(nn.CrossEntropyLoss(), outputs, labels_a, labels_b, lam)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += (lam * predicted.eq(labels_a).float() + 
                                  (1 - lam) * predicted.eq(labels_b).float()).sum().item()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            progress_bar.set_postfix({'loss': loss.item(), 
                                      'acc': 100. * train_correct / train_total,
                                      'lr': optimizer.param_groups[0]["lr"]})
        
        epoch_loss = running_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                if use_mixup:
                    outputs = model(inputs)
                    if use_mixup:
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]["lr"])
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'learning_rates': learning_rates
            }, f'./checkpoint/best_cnn_model_{name}_{loss_type}.pth')
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
              f'Val loss: {val_loss:.4f}, '
              f'Val acc: {val_acc:.4f}, Best val acc: {best_val_acc:.4f}')
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title(f'Loss Curves - {loss_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.title(f'Accuracy Curves - {loss_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(learning_rates)
    plt.title(f'Learning Rate - {loss_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(f'./graph/loss/training_curves_{name}_{loss_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    checkpoint = torch.load(f'./checkpoint/best_cnn_model_{name}_{loss_type}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}, validation accuracy: {checkpoint['best_val_acc']:.4f}")
    
    return model, checkpoint['train_losses'], checkpoint['val_losses'], checkpoint['train_accs'], checkpoint['val_accs']

def evaluate_model(model, test_loader, device, class_names=None, name="model", loss_type="ce"):
    model.eval()
    test_preds = []
    test_true = []
    all_probs = []
    
    inference_times = []
    
    num_classes = 22
    class_correct_top1 = [0] * num_classes
    class_correct_top5 = [0] * num_classes
    class_total = [0] * num_classes
    class_incorrect = [0] * num_classes
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            
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
                else:
                    class_incorrect[label] += 1
                if correct_top5[i]:
                    class_correct_top5[label] += 1
            
            test_preds.extend(top1_preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    all_probs = np.vstack(np.concatenate(all_probs))
    
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    
    class_accuracy_top1 = [class_correct_top1[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    class_accuracy_top5 = [class_correct_top5[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    
    mean_class_accuracy = sum(class_accuracy_top1) / num_classes
    
    cm = confusion_matrix(test_true, test_preds)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    inference_per_sample = avg_inference_time / test_loader.batch_size
    
    result_file = f'./results/{name}_{loss_type}_evaluation.txt'
    with open(result_file, 'w') as f:
        header = "===== Model Evaluation Report =====\n"
        f.write(header)
        print(header)
        
        overall = f"Top-1 Accuracy: {top1_accuracy:.4f}\n"
        overall += f"Top-5 Accuracy: {top5_accuracy:.4f}\n"
        overall += f"Mean Class Accuracy: {mean_class_accuracy:.4f}\n"
        overall += f"Average batch inference time: {avg_inference_time*1000:.2f} ms\n"
        overall += f"Average per-sample inference time: {inference_per_sample*1000:.2f} ms\n"
        f.write(overall)
        print(overall)
        
        f.write("\nDetailed class performance:\n")
        print("\nDetailed class performance:")
        
        class_header = f"{'Class':^30} | {'Top-1 Acc':^12} | {'Top-5 Acc':^12} | {'Correct':^8} | {'Incorrect':^8} | {'Total':^10}\n"
        f.write(class_header)
        f.write('-' * len(class_header) + '\n')
        print(class_header)
        print('-' * len(class_header))
        
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f"Class {i}"
            class_row = f"{class_name:^30} | {class_accuracy_top1[i]:^12.4f} | {class_accuracy_top5[i]:^12.4f} | {class_correct_top1[i]:^8} | {class_incorrect[i]:^8} | {class_total[i]:^10}\n"
            f.write(class_row)
            print(class_row)
        
        best_class_idx = np.argmax(class_accuracy_top1)
        worst_class_idx = np.argmin(class_accuracy_top1)
        
        best_class = f"\nBest performing class: {class_names[best_class_idx] if class_names else best_class_idx}"
        best_class += f" (Accuracy: {class_accuracy_top1[best_class_idx]:.4f})\n"
        
        worst_class = f"Worst performing class: {class_names[worst_class_idx] if class_names else worst_class_idx}"
        worst_class += f" (Accuracy: {class_accuracy_top1[worst_class_idx]:.4f})\n"
        
        f.write(best_class)
        f.write(worst_class)
        print(best_class)
        print(worst_class)
    
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    plt.rcParams['axes.facecolor'] = '#eef1f5'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 1.3
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12
    
    plt.figure(figsize=(20, 16))
    sns.set(font_scale=1.2)
    cm_display = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                             xticklabels=class_names if class_names else range(num_classes),
                             yticklabels=class_names if class_names else range(num_classes))
    plt.xlabel('Predicted Class', fontsize=15)
    plt.ylabel('True Class', fontsize=15)
    plt.title(f'Confusion Matrix (Top-1: {top1_accuracy:.4f}, Top-5: {top5_accuracy:.4f}, Mean: {mean_class_accuracy:.4f})', fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'./graph/cm/cm_{name}_{loss_type}_{top1_accuracy:.4f}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    report = classification_report(test_true, test_preds, target_names=class_names if class_names else [str(i) for i in range(num_classes)], output_dict=True)
    
    f1_scores = [report[class_names[i] if class_names else str(i)]['f1-score'] for i in range(num_classes)]
    
    plt.figure(figsize=(14, 6))
    ax = plt.axes()
    ax.set_facecolor('#eef1f5')
    bars = plt.bar(range(num_classes), f1_scores, color='steelblue')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{f1_scores[i]:.3f}', ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Scores by Class', fontsize=14)
    plt.xticks(range(num_classes), class_names if class_names else range(num_classes), rotation=90)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='-', color='white', linewidth=1.3)
    plt.tight_layout()
    plt.savefig(f'./graph/others/f1_{name}_{loss_type}_{top1_accuracy:.4f}.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(14, 6))
    ax = plt.axes()
    ax.set_facecolor('#eef1f5')
    
    categories = list(range(num_classes))
    
    top1_values = class_accuracy_top1
    top5_values = class_accuracy_top5
    mean_line = [mean_class_accuracy] * num_classes
    
    bar_width = 0.35
    index = np.arange(len(categories))
    
    plt.bar(index, top1_values, bar_width, label='Top-1 Accuracy', color='steelblue')
    plt.bar(index + bar_width, top5_values, bar_width, label='Top-5 Accuracy', color='lightcoral')
    
    plt.plot([-0.5, num_classes-0.5], [mean_class_accuracy, mean_class_accuracy], 
             'k--', label=f'Mean Class Accuracy: {mean_class_accuracy:.4f}')
    
    plt.plot([-0.5, num_classes-0.5], [top1_accuracy, top1_accuracy], 
             'b--', label=f'Overall Top-1 Accuracy: {top1_accuracy:.4f}')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Comparison by Class', fontsize=14)
    plt.xticks(index + bar_width/2, class_names if class_names else categories, rotation=90)
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    plt.grid(axis='both', linestyle='-', color='white', linewidth=1.3)
    plt.tight_layout()
    plt.savefig(f'./graph/acc/acc_compare_{name}_{loss_type}_{top1_accuracy:.4f}.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(14, 6))
    ax = plt.axes()
    ax.set_facecolor('#eef1f5')
    
    bars = plt.bar(range(num_classes), class_total, color='steelblue')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{class_total[i]}', ha='center', va='bottom', rotation=90, fontsize=8)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.title('Sample Distribution by Class', fontsize=14)
    plt.xticks(range(num_classes), class_names if class_names else range(num_classes), rotation=90)
    plt.grid(axis='y', linestyle='-', color='white', linewidth=1.3)
    plt.tight_layout()
    plt.savefig(f'./graph/others/class_distribution_{name}_{loss_type}.png', dpi=300)
    plt.close()
    
    return top1_accuracy, top5_accuracy, mean_class_accuracy, cm

def main():
    train_dataset_raw = torch.load('./data/train_data_mgr_only.pt')
    val_dataset_raw = torch.load('./data/val_data_mgr_only.pt')
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 64
    num_classes = 22
    
    class_names = get_class_names()
    
    train_loader = DataLoader(
        train_dataset_raw, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset_raw, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    sample_data = train_dataset_raw[0][0]
    print(f"Data shape: {sample_data.shape}")
    
    results_summary = []
    
    loss_types = ['ce','ce']
    
    for loss_type in loss_types:
        print(f"\n==== Loss function: {loss_type} ====")
        
        model = GaussianCNN(
            num_features=sample_data.shape[0],
            num_frames=sample_data.shape[1],
            channels=sample_data.shape[2],
            num_classes=num_classes
        )
        model.to(device)
        
        if loss_type == 'ce':
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, device, 
            num_epochs=200, 
            loss_type=loss_type,
            weight_decay=0.0003, 
            name=f'cnn_interleaved'
        )
        
        top1_accuracy, top5_accuracy, mean_class_accuracy, cm = evaluate_model(
            model, val_loader, device, class_names, name=f'cnn_interleaved', loss_type=loss_type
        )
        
        results_summary.append({
            'loss_type': loss_type,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'mean_class_accuracy': mean_class_accuracy
        })
        
        print(f"Loss function {loss_type} evaluation results:")
        print(f"  Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
        print(f"  Mean Class Accuracy: {mean_class_accuracy:.4f}")
    
    print("\n===== Loss Function Performance Comparison =====")
    print(f"{'Loss Function':^15} | {'Top-1 Acc':^12} | {'Top-5 Acc':^12} | {'Mean Class Acc':^16}")
    print('-' * 61)
    
    with open('./results/loss_comparison.txt', 'w') as f:
        f.write("===== Loss Function Performance Comparison =====\n")
        f.write(f"{'Loss Function':^15} | {'Top-1 Acc':^12} | {'Top-5 Acc':^12} | {'Mean Class Acc':^16}\n")
        f.write('-' * 61 + '\n')
        
        for result in results_summary:
            row = f"{result['loss_type']:^15} | {result['top1_accuracy']:^12.4f} | {result['top5_accuracy']:^12.4f} | {result['mean_class_accuracy']:^16.4f}"
            print(row)
            f.write(row + '\n')
    
    plt.figure(figsize=(12, 8))
    
    loss_names = [r['loss_type'] for r in results_summary]
    top1_values = [r['top1_accuracy'] for r in results_summary]
    top5_values = [r['top5_accuracy'] for r in results_summary]
    mean_values = [r['mean_class_accuracy'] for r in results_summary]
    
    x = np.arange(len(loss_names))
    width = 0.25
    
    plt.bar(x - width, top1_values, width, label='Top-1 Accuracy', color='steelblue')
    plt.bar(x, top5_values, width, label='Top-5 Accuracy', color='lightcoral')
    plt.bar(x + width, mean_values, width, label='Mean Class Accuracy', color='forestgreen')
    
    for i, v in enumerate(top1_values):
        plt.text(i - width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    for i, v in enumerate(top5_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    for i, v in enumerate(mean_values):
        plt.text(i + width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Loss Function Performance Comparison', fontsize=14)
    plt.xticks(x, loss_names)
    plt.ylim(0, max(max(top1_values), max(top5_values), max(mean_values)) + 0.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./graph/others/loss_function_comparison.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    main()