import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ..src.utils import *
from ..src.models.model import GaussianCNN
from ..tools.train import train_model
from ..tools.evaluation import evaluate_model

class JointOnlyDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        data, label = self.original_dataset[idx]
        
        num_bones = (data.shape[0] - 1) // 2
        num_joints = num_bones + 1
        
        joint_only_data = torch.zeros((num_joints, data.shape[1], data.shape[2]), device=data.device)
        
        joint_only_data[0] = data[0]
        
        for i in range(1, num_joints):
            joint_only_data[i] = data[i*2 - 1] 
            
        return joint_only_data, label


def main():
    print("Loading training and validation datasets...")
    train_dataset = torch.load('data/train_data_bone_joint_cross_full.pt')
    val_dataset = torch.load('data/val_data_bone_joint_cross_full.pt')
    
    print("Extracting joint data...")
    train_joint_dataset = JointOnlyDataset(train_dataset)
    val_joint_dataset = JointOnlyDataset(val_dataset)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    sample_data, _ = train_joint_dataset[0]
    num_joints = sample_data.shape[0]
    print(f"Number of joints: {num_joints}")
    
    batch_size = 64
    gmm_components = 22
    gaussian_params = 10
    num_classes = 22
    
    model = GaussianCNN(num_joints=num_joints, gmm_components=gmm_components, 
                         gaussian_params=gaussian_params, num_classes=num_classes)
    model.to(device)
    
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    train_loader = DataLoader(
        train_joint_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_joint_dataset,
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print("Starting model training...")
    model, train_losses, val_accs = train_model(
        model, train_loader, val_loader, device, num_epochs=200, name='joint_only'
    )
    
    class_names = get_class_names()
    
    print("Evaluating model...")
    top1_accuracy, top5_accuracy, cm = evaluate_model(model, val_loader, device, class_names, name='joint_only')
    print(f"Final results - Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}")

if __name__ == "__main__":
    os.makedirs('./checkpoint', exist_ok=True)
    os.makedirs('./graph/cm', exist_ok=True)
    os.makedirs('./graph/others', exist_ok=True)
    
    main()