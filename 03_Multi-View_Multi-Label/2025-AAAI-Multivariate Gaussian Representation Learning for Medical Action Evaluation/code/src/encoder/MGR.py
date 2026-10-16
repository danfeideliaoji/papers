import numpy as np
import torch
import json
import os
from torch.utils.data import Dataset,DataLoader
import sys
import time
from sklearn.mixture import GaussianMixture

class CPRDataLoader_with_pt(Dataset):
    def __init__(self, data_root: str, 
                 split_file: str,
                 phase: str = 'train',
                 max_frames: int = 150,
                 num_joints: int = 133,
                 center_idx: int = 0,
                 dataset=None,
                 data=None,
                 gmm_components: int = 22,
                 use_cuda: bool = True,
                 device=None):
        self.data_root = data_root
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.center_idx = center_idx
        self.gmm_components = gmm_components
        self.use_cuda = use_cuda
        self.device = device
        
        self.key_joint_indices = list(range(68)) 
        
        self.dataset = dataset
        self.data = []
        
        start_time = time.time()
        print(f"Start processing {len(self.dataset)} samples...")
        
        for i in range(len(self.dataset)):
            video_data = self.dataset[i][0]
            label = self.dataset[i][1]
            data_processed = self.preprocess(video_data)
            self.data.append((data_processed, label))
            if (i+1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i+1} samples ({elapsed:.2f} seconds)")
                
        print(f"Finished processing {len(self.dataset)} samples in {time.time() - start_time:.2f} seconds") 

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def preprocess(self, raw_data):
        frames = raw_data
        
        T0 = len(frames)
        full_data = np.zeros((T0, self.num_joints, 3))
        if isinstance(frames, torch.Tensor):
            full_data = frames.numpy()
        else:
            full_data = frames
        
        trunk_data = np.zeros((T0, len(self.key_joint_indices), 3))
        trunk_data[:,:,:2] = full_data[:, self.key_joint_indices, :]
        
        T = trunk_data.shape[0]
        J = len(self.key_joint_indices)
        fps = 25
        time_scale = 400
        
        t_vals = (np.arange(T) / fps) * time_scale
        for t in range(T):
            trunk_data[t, :, 2] = t_vals[t]
        
        K = self.gmm_components
        
        gaussian_params = np.zeros((J, K, 10))
        
        for j in range(J):
            joint_trajectory = trunk_data[:, j, :]
                        
            device = self.device
            gmm = GaussianMixture(
                n_components=K,
                covariance_type='full',
                init_params='kmeans',
                random_state=42
            )

            gmm.fit(joint_trajectory)
            
            for k in range(K):
                mean = gmm.means_[k]
                gaussian_params[j, k, :3] = mean
                
                covariance = gmm.covariances_[k]
                
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                
                lambdas = np.sqrt(np.abs(eigenvalues))
                gaussian_params[j, k, 3:6] = lambdas
                
                rotation_matrix = eigenvectors
                
                quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
                gaussian_params[j, k, 6:] = quaternion
        
        return torch.from_numpy(gaussian_params).float()
   
    def rotation_matrix_to_quaternion(self, R):
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            
        q = np.array([qw, qx, qy, qz])
        q = q / np.linalg.norm(q)
        
        return q