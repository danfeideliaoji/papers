import numpy as np
import torch
from torch.utils.data import Dataset


def get_class_names():
    return [
        "Correct",                    # 0
        "Depth-Insufficient Press",   # 1
        "Depth-Excess",               # 2
        "Depth-Insufficient Rebound", # 3
        "Freq-High Frequency",        # 4
        "Freq-Low Frequency",         # 5
        "Freq-Excessive Momentum",    # 6
        "Position-Wrong Position",    # 7
        "Position-Movement",          # 8
        "Hand-Double Hand-Cross",     # 9
        "Hand-Double Hand-Overlap",   # 10
        "Hand-Double Hand-Two Fist",  # 11
        "Hand-One Hand-Fist",         # 12
        "Hand-One Hand-Hand",         # 13
        "Angle-Arm Bend",             # 14
        "Angle-Left Skew",            # 15
        "Angle-Right Skew",           # 16
        "Angle-Front Skew",           # 17
        "Angle-Back Skew",            # 18
        "Body-Waist Unstable",        # 19
        "Body-Thigh Antevert",        # 20
        "Body-Thigh Tilt Back"        # 21
    ]

def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class SkeletonAugmentation:
    def __init__(self):
        pass
        
    def add_gaussian_noise(self, data, noise_level=0.3):
        noise = torch.randn_like(data) * noise_level
        return data + noise
    
    def random_mask(self, data, mask_prob=0.2):
        mask = torch.rand_like(data[:, :, 0:1]) > mask_prob
        mask = mask.expand_as(data)
        return data * mask
    
    def scale_augmentation(self, data, scale_range=(0.8, 1.2)):
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        data_scaled = data.clone()
        data_scaled[:, :, :6] = data[:, :, :6] * scale_factor
        return data_scaled
    
    def time_shift(self, data, max_shift=0.5):
        shift = np.random.uniform(-max_shift, max_shift)
        data_shifted = data.clone()
        data_shifted[:, :, 2] = data[:, :, 2] + shift
        return data_shifted
    
    def jitter(self, data, jitter_scale=0.1):
        jitter = torch.randn_like(data) * jitter_scale
        return data + jitter
        
    def __call__(self, data, augment_prob=0.5):
        if np.random.random() > augment_prob:
            return data
            
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
        
        augmented_data = data.clone()
        for method_idx in chosen_methods:
            method = aug_methods[method_idx][0]
            augmented_data = method(augmented_data)
            
        return augmented_data

class RepeatedDataset(Dataset):
    def __init__(self, dataset, repeat_times=5, apply_augmentation=True):
        self.dataset = dataset
        self.repeat_times = repeat_times
        self.augmentation = SkeletonAugmentation() if apply_augmentation else None
        
        self.total_size = len(dataset) * repeat_times
        
        print(f"Original dataset size: {len(dataset)}")
        print(f"Dataset size after {repeat_times} repetitions: {self.total_size}")
        
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        original_idx = idx % len(self.dataset)
        data, label = self.dataset[original_idx]
        
        if idx >= len(self.dataset) and self.augmentation is not None:
            aug_intensity = 0.05 + (idx // len(self.dataset)) * 0.11
            data = self.augmentation(data, augment_prob=0.75)
            
        return data, label
    
