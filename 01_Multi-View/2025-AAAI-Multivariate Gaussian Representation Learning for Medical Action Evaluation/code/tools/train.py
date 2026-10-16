import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from utils import *
from ..src.models.model import GaussianCNN
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import os


def train_model(model, train_loader, val_loader, device, num_epochs=100, smoothing=0.13, 
                weight_decay=1e-4, scheduler_patience=8, name='', lr=0.005, mixup_alpha=0.2):
    print("Starting training process...")
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=scheduler_patience, 
        verbose=True, threshold=0.0005, min_lr=1e-6
    )
    
    best_val_acc = 0.0
    patience = 1000
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
            
            # Apply mixup
            mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha, device=device)
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # For tracking accuracy with mixup
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (lam * predicted.eq(labels_a).float() + 
                             (1 - lam) * predicted.eq(labels_b).float()).sum().item()
            
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
              f'Train loss: {epoch_loss:.4f}, Train accuracy: {train_acc:.4f}, '
              f'Validation loss: {val_loss/val_total:.4f}, '
              f'Validation accuracy: {val_acc:.4f}, Best validation accuracy: {best_val_acc:.4f}')
    
    checkpoint = torch.load(f'./checkpoint/best_cnn_model_{name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}, validation accuracy: {checkpoint['best_val_acc']:.4f}")
    
    return model, train_losses, val_accs


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train GaussianCNN model with various options')
    
    # Dataset and model parameters
    parser.add_argument('--train_data', type=str, default='data/train_data_bone_joint_cross_full.pt',
                        help='Path to the training dataset')
    parser.add_argument('--val_data', type=str, default='data/val_data_bone_joint_cross_full.pt',
                        help='Path to the validation dataset')
    parser.add_argument('--num_joints', type=int, default=117,
                        help='Number of joints in the input data')
    parser.add_argument('--gmm_components', type=int, default=22,
                        help='Number of GMM components')
    parser.add_argument('--gaussian_params', type=int, default=10,
                        help='Number of Gaussian parameters')
    parser.add_argument('--num_classes', type=int, default=22,
                        help='Number of output classes')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--scheduler_patience', type=int, default=8,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Alpha parameter for mixup augmentation')
    parser.add_argument('--smoothing', type=float, default=0.13,
                        help='Label smoothing parameter')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda:0", "cpu"). If None, will use CUDA if available.')
    parser.add_argument('--name', type=str, default='joint_only',
                        help='Name suffix for saving the best model')
    
    # Parse the arguments
    args = parser.parse_args()
    
    print(f"Loading training dataset from {args.train_data}...")
    train_dataset = torch.load(args.train_data)
    
    print(f"Loading validation dataset from {args.val_data}...")
    val_dataset = torch.load(args.val_data)
    
    print("Extracting joint data...")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = GaussianCNN(num_joints=args.num_joints, 
                        gmm_components=args.gmm_components, 
                        gaussian_params=args.gaussian_params, 
                        num_classes=args.num_classes)
    model.to(device)
    
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    print("Starting model training...")
    model, train_losses, val_accs = train_model(
        model, 
        train_loader, 
        val_loader, 
        device, 
        num_epochs=args.num_epochs, 
        smoothing=args.smoothing,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        name=args.name,
        lr=args.lr,
        mixup_alpha=args.mixup_alpha
    )
    


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('./checkpoint', exist_ok=True)
    os.makedirs('./graph/cm', exist_ok=True)
    os.makedirs('./graph/others', exist_ok=True)
    
    main()