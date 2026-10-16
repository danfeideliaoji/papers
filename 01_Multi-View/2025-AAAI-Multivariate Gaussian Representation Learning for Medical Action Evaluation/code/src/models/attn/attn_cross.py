import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch.nn.functional as F
from ..utils import mixup_data, mixup_criterion, get_class_names

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.3, proj_drop=0.3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        x = x + x_attn
        
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.3):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x_norm = self.norm(x)
        return x + self.net(x_norm)


class JointBoneProcessor(nn.Module):
    def __init__(self, input_dim=10, feature_dim=64, num_heads=8, dropout=0.3, num_layers=2):
        super().__init__()
        
        self.joint_embed = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        self.bone_embed = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        self.joint_attn_layers = nn.ModuleList([])
        self.joint_ffn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.joint_attn_layers.append(
                SelfAttention(
                    dim=feature_dim,
                    num_heads=num_heads,
                    attn_drop=dropout,
                    proj_drop=dropout
                )
            )
            self.joint_ffn_layers.append(
                FeedForward(dim=feature_dim, dropout=dropout)
            )
        
        self.bone_attn_layers = nn.ModuleList([])
        self.bone_ffn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.bone_attn_layers.append(
                SelfAttention(
                    dim=feature_dim,
                    num_heads=num_heads,
                    attn_drop=dropout,
                    proj_drop=dropout
                )
            )
            self.bone_ffn_layers.append(
                FeedForward(dim=feature_dim, dropout=dropout)
            )
        
        self.joint_final_norm = nn.LayerNorm(feature_dim)
        self.bone_final_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, joint_data, bone_data):
        batch_size = joint_data.shape[0]
        num_joints = joint_data.shape[1]
        num_bones = bone_data.shape[1]
        num_components = joint_data.shape[2]
        
        joint_flat = joint_data.reshape(batch_size, num_joints * num_components, -1)
        bone_flat = bone_data.reshape(batch_size, num_bones * num_components, -1)
        
        joint_feats = self.joint_embed(joint_flat)
        bone_feats = self.bone_embed(bone_flat)
        
        for attn_layer, ffn_layer in zip(self.joint_attn_layers, self.joint_ffn_layers):
            joint_feats = attn_layer(joint_feats)
            joint_feats = ffn_layer(joint_feats)
        
        for attn_layer, ffn_layer in zip(self.bone_attn_layers, self.bone_ffn_layers):
            bone_feats = attn_layer(bone_feats)
            bone_feats = ffn_layer(bone_feats)
        
        joint_feats = self.joint_final_norm(joint_feats)
        bone_feats = self.bone_final_norm(bone_feats)
        
        joint_reshaped = joint_feats.reshape(batch_size, num_joints, num_components, -1)
        bone_reshaped = bone_feats.reshape(batch_size, num_bones, num_components, -1)
        
        interleaved_features = []
        
        interleaved_features.append(joint_reshaped[:, 0:1, :, :])
        
        for i in range(num_bones):
            interleaved_features.append(bone_reshaped[:, i:i+1, :, :])
            if i+1 < num_joints:
                interleaved_features.append(joint_reshaped[:, i+1:i+2, :, :])
        
        interleaved = torch.cat(interleaved_features, dim=1)
        
        return interleaved


class SelfAttentionCNN(nn.Module):
    def __init__(self, num_joints=21, num_bones=20, gmm_components=25, 
                 input_dim=10, feature_dim=64, num_heads=8, num_classes=22, 
                 dropout=0.3, num_attn_layers=2):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_bones = num_bones
        self.gmm_components = gmm_components
        self.feature_dim = feature_dim
        
        self.processor = JointBoneProcessor(
            input_dim=input_dim,
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_attn_layers
        )
        
        self.pre_cnn_norm = nn.LayerNorm([num_joints + num_bones, gmm_components, feature_dim])
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        total_items = num_joints + num_bones
        h_out = total_items // 2 // 2 // 2
        w_out = gmm_components // 2 // 2 // 2
        h_out = max(1, h_out)
        w_out = max(1, w_out)
        
        self.fc_input_size = 256 * h_out * w_out
        print(f"Linear layer input size: {self.fc_input_size}")
        
        self.pre_classifier_norm = nn.LayerNorm(self.fc_input_size)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.fc_input_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data_tuple, return_features=False):
        joint_data, bone_data = data_tuple
        batch_size = joint_data.size(0)
        
        processed_data = self.processor(joint_data, bone_data)
        
        processed_data = self.pre_cnn_norm(processed_data)
        
        x = processed_data.permute(0, 3, 1, 2)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        global_feat = self.global_pool(x)
        global_feat = global_feat.view(batch_size, -1)
        global_logits = self.global_fc(global_feat)
        
        x_flat = x.reshape(batch_size, -1)
        x_norm = self.pre_classifier_norm(x_flat)
        main_logits = self.classifier(x_norm)
        
        logits = main_logits + 0.2 * global_logits
        
        if return_features:
            return logits, x_flat
        return logits


class JointBoneDataProcessor:
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        sample_data, sample_label = self.train_dataset[0]
        
        print(f"Original data shape: {sample_data.shape}")
        print(f"Number of training samples: {len(self.train_dataset)}")
        print(f"Number of validation samples: {len(self.val_dataset)}")
        
    def separate_joint_bone_data(self, remove_first_joint=False):
        processed_train = []
        processed_val = []
        
        for data, label in self.train_dataset:
            joint_data, bone_data = self._separate_single_sample(data, remove_first_joint)
            processed_train.append(((joint_data, bone_data), label))
        
        for data, label in self.val_dataset:
            joint_data, bone_data = self._separate_single_sample(data, remove_first_joint)
            processed_val.append(((joint_data, bone_data), label))
            
        print(f"Processed train dataset size: {len(processed_train)}")
        print(f"Processed val dataset size: {len(processed_val)}")
        
        if len(processed_train) > 0:
            sample = processed_train[0]
            joint_shape = sample[0][0].shape
            bone_shape = sample[0][1].shape
            print(f"Joint data shape: {joint_shape}")
            print(f"Bone data shape: {bone_shape}")
        
        return processed_train, processed_val
    
    def _separate_single_sample(self, data, remove_first_joint):
        total_elements = data.shape[0]
        
        joint_indices = []
        bone_indices = []
        
        joint_indices.append(0)
        
        for i in range(1, total_elements):
            if i % 2 == 1:
                joint_indices.append(i)
            else:
                bone_indices.append(i)
        
        joint_data = data[joint_indices]
        bone_data = data[bone_indices]
        
        if remove_first_joint and len(joint_data) > 0:
            joint_data = joint_data[1:]
        
        return joint_data, bone_data

    def create_dataloaders(self, batch_size=32, num_workers=4):
        processed_train, processed_val = self.separate_joint_bone_data()
        
        def custom_collate_fn(batch):
            joint_data = torch.stack([item[0][0] for item in batch])
            bone_data = torch.stack([item[0][1] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
            return (joint_data, bone_data), labels
        
        train_loader = DataLoader(
            processed_train, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
        )
        
        val_loader = DataLoader(
            processed_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
        )
        
        return train_loader, val_loader


class SkeletonAugmentation:
    def __init__(self):
        pass
        
    def add_gaussian_noise(self, data, noise_level=0.3):
        noise = torch.randn_like(data) * noise_level
        return data + noise
    
    def random_mask(self, data, mask_prob=0.15):
        mask = torch.rand_like(data[:, :, 0:1]) > mask_prob
        mask = mask.expand_as(data)
        return data * mask
    
    def scale_augmentation(self, data, scale_range=(0.9, 1.1)):
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        data_scaled = data.clone()
        data_scaled[:, :, :6] = data[:, :, :6] * scale_factor
        return data_scaled
    
    def time_shift(self, data, max_shift=0.1):
        shift = np.random.uniform(-max_shift, max_shift)
        data_shifted = data.clone()
        data_shifted[:, :, 2] = data[:, :, 2] + shift
        return data_shifted
    
    def jitter(self, data, jitter_scale=0.1):
        jitter = torch.randn_like(data) * jitter_scale
        return data + jitter
        
    def __call__(self, data_tuple, augment_prob=0.5):
        joint_data, bone_data = data_tuple
        
        if np.random.random() > augment_prob:
            return joint_data, bone_data
            
        num_augs = np.random.randint(1, 4)
        
        aug_methods = [
            (self.add_gaussian_noise, 0.3),
            (self.random_mask, 0.2),
            (self.scale_augmentation, 0.2),
            (self.time_shift, 0.15),
            (self.jitter, 0.15)
        ]
        
        chosen_methods = np.random.choice(
            [i for i in range(len(aug_methods))],
            size=num_augs,
            replace=False,
            p=[w for _, w in aug_methods]
        )
        
        joint_data_aug = joint_data.clone()
        bone_data_aug = bone_data.clone()
        
        for method_idx in chosen_methods:
            method = aug_methods[method_idx][0]
            joint_data_aug = method(joint_data_aug)
            bone_data_aug = method(bone_data_aug)
            
        return joint_data_aug, bone_data_aug


class RepeatedJointBoneDataset(Dataset):
    def __init__(self, dataset, repeat_times=5, apply_augmentation=True):
        self.dataset = dataset
        self.repeat_times = repeat_times
        self.augmentation = SkeletonAugmentation() if apply_augmentation else None
        
        self.total_size = len(dataset) * repeat_times
        
        print(f"Original dataset size: {len(dataset)}")
        print(f"After {repeat_times} repetitions, dataset size: {self.total_size}")
        
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        original_idx = idx % len(self.dataset)
        data_tuple, label = self.dataset[original_idx]
        
        if idx >= len(self.dataset) and self.augmentation is not None:
            aug_intensity = 0.05 + (idx // len(self.dataset)) * 0.03
            data_tuple = self.augmentation(data_tuple, augment_prob=0.75)
            
        return data_tuple, label


def train_model(model, train_loader, val_loader, device, num_epochs=100, 
                weight_decay=1e-4, scheduler_patience=8, name='', lr=0.005, mixup_alpha=0.2):
    print("Starting training process...")
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=scheduler_patience, 
        verbose=True, threshold=0.001, min_lr=1e-6
    )
    
    best_val_acc = 0.0
    patience = 100
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
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                joint_data, bone_data = inputs
                joint_data = joint_data.to(device)
                bone_data = bone_data.to(device)
                inputs = (joint_data, bone_data)
            else:
                inputs = inputs.to(device)
                
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha, device=device)
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (lam * predicted.eq(labels_a).float() + 
                             (1 - lam) * predicted.eq(labels_b).float()).sum().item()
            
            loss.backward()
            optimizer.step()
            
            if isinstance(inputs, tuple):
                batch_size = inputs[0].size(0)
            else:
                batch_size = inputs.size(0)
                
            running_loss += loss.item() * batch_size
            
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
                if isinstance(inputs, tuple) or isinstance(inputs, list):
                    joint_data, bone_data = inputs
                    joint_data = joint_data.to(device)
                    bone_data = bone_data.to(device)
                    inputs = (joint_data, bone_data)
                else:
                    inputs = inputs.to(device)
                    
                labels = labels.to(device)
                
                outputs = model(inputs)
                
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                if isinstance(inputs, tuple):
                    batch_size = inputs[0].size(0)
                else:
                    batch_size = inputs.size(0)
                    
                val_loss += loss.item() * batch_size
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_name = f'./checkpoint/best_self_attn_cnn_model_{name}_mixup{mixup_alpha}.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_name)
            patience_counter = 0
            print(f'Validation accuracy improved to {best_val_acc:.4f}, model saved.')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
            
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss/val_total:.4f}, '
              f'Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}')
    
    best_model_path = f'./checkpoint/best_self_attn_cnn_model_{name}_mixup{mixup_alpha}.pth'
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}, Val Acc: {checkpoint['best_val_acc']:.4f}")
    
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
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                joint_data, bone_data = inputs
                joint_data = joint_data.to(device)
                bone_data = bone_data.to(device)
                inputs = (joint_data, bone_data)
            else:
                inputs = inputs.to(device)
            
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
    plt.savefig(f'./graph/cm/confusion_matrix_self_attn_{top1_accuracy:.4f}_{top5_accuracy:.4f}.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'./graph/others/f1_scores_self_attn_{top1_accuracy:.4f}_{top5_accuracy:.4f}.png', dpi=300)
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
    plt.savefig(f'./graph/others/top1_vs_top5_accuracy_self_attn_{top1_accuracy:.4f}_{top5_accuracy:.4f}.png', dpi=300)
    plt.close()
    
    return top1_accuracy, top5_accuracy, cm


def main():
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_dataset = torch.load('./data/train_data_bone_joint_cross_full.pt')
    val_dataset = torch.load('./data/val_data_bone_joint_cross_full.pt')
    
    processor = JointBoneDataProcessor(train_dataset, val_dataset)
    
    print("Creating dataloaders...")
    train_loader, val_loader = processor.create_dataloaders(batch_size=32)
    
    print("Applying data augmentation...")
    augmented_train_dataset = RepeatedJointBoneDataset(
        processor.separate_joint_bone_data()[0], 
        repeat_times=1,
        apply_augmentation=True
    )
    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )
    
    for (joint_data, bone_data), labels in train_loader:
        print(f"Joint data shape: {joint_data.shape}")
        print(f"Bone data shape: {bone_data.shape}")
        print(f"Labels shape: {labels.shape}")
        
        num_joints = joint_data.shape[1]
        num_bones = bone_data.shape[1]
        num_components = joint_data.shape[2]
        break
    
    print("Initializing model...")
    model = SelfAttentionCNN(
        num_joints=num_joints,
        num_bones=num_bones,
        gmm_components=num_components,
        input_dim=10,
        feature_dim=256,
        num_heads=8,
        num_classes=22,
        dropout=0.5,
        num_attn_layers=1
    )
    model = model.to(device)
    
    print("Starting training...")
    model, train_losses, val_accs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=200,
        weight_decay=3e-4,
        scheduler_patience=5,
        name='lr0005_dim256_heads8_selfattn1',
        lr=0.005,
        mixup_alpha=0.2
    )
    
    print("Evaluating model...")
    class_names = [f"Class {i}" for i in range(22)]
    top1_acc, top5_acc, confusion_mat = evaluate_model(
        model=model,
        test_loader=val_loader,
        device=device,
        class_names=class_names
    )
    
    print(f"Final Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Final Top-5 Accuracy: {top5_acc:.4f}")


if __name__ == "__main__":
    main()