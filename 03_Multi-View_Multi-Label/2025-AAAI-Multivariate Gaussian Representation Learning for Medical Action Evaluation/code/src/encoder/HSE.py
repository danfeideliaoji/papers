import numpy as np
import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
import time
import math
from sklearn.mixture import GaussianMixture

class CPRDataLoader_JointBone_Interleaved(Dataset):
    def __init__(self, data_root: str, 
                 phase: str = 'train',
                 max_frames: int = 150,
                 num_joints: int = 133,
                 center_idx: int = 0,
                 dataset=None,
                 data=None,
                 gmm_components: int = 25,
                 use_cuda: bool = True,
                 device=None):
        self.data_root = data_root
        self.max_frames = max_frames
        self.num_joints = num_joints
        self.center_idx = center_idx
        self.gmm_components = gmm_components
        self.use_cuda = use_cuda
        self.device = device

        self.key_joint_indices = list(range(17)) + list(range(91, 133))
        
        self.connections = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
            (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
            (1, 0), (3, 1), (2, 0), (4, 2), (91, 9), (92, 91), (93, 92), (94, 93), 
            (95, 94), (96, 91), (97, 96), (98, 97), (99, 98), (100, 91), (101, 100), 
            (102, 101), (103, 102), (104, 91), (105, 104), (106, 105), (107, 106), 
            (108, 91), (109, 108), (110, 109), (111, 110), (112, 10), (113, 112), 
            (114, 113), (115, 114), (116, 115), (117, 112), (118, 117), (119, 118), 
            (120, 119), (121, 112), (122, 121), (123, 122), (124, 123), (125, 112), 
            (126, 125), (127, 126), (128, 127), (129, 112), (130, 129), (131, 130), (132, 131)
        ]
        
        self.connection_indices = []
        for a, b in self.connections:
            if a in self.key_joint_indices and b in self.key_joint_indices:
                idx_a = self.key_joint_indices.index(a)
                idx_b = self.key_joint_indices.index(b)
                self.connection_indices.append((idx_a, idx_b))
        
        self.bone_to_joint_map = {}
        for i, (a_idx, _) in enumerate(self.connection_indices):
            if a_idx not in self.bone_to_joint_map:
                self.bone_to_joint_map[a_idx] = []
            self.bone_to_joint_map[a_idx].append(i)
        
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
        
        trunk_data = full_data[:, self.key_joint_indices, :]
        
        T = trunk_data.shape[0]
        J = len(self.key_joint_indices)
        fps = 25
        time_scale = 400
        
        t_vals = (np.arange(T) / fps) * time_scale
        for t in range(T):
            trunk_data[t, :, 2] = t_vals[t]
        
        K = self.gmm_components
        
        joint_gaussian_params = np.zeros((J, K, 10))
        
        for j in range(J):
            joint_trajectory = trunk_data[:, j, :]
            
            gmm = GaussianMixture(
                n_components=K,
                covariance_type='full',
                init_params='kmeans',
                random_state=42
            )

            gmm.fit(joint_trajectory)
            
            for k in range(K):
                mean = gmm.means_[k]
                joint_gaussian_params[j, k, :3] = mean
                
                covariance = gmm.covariances_[k]
                
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                
                lambdas = np.sqrt(np.abs(eigenvalues))
                joint_gaussian_params[j, k, 3:6] = lambdas
                
                rotation_matrix = eigenvectors
                
                quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
                joint_gaussian_params[j, k, 6:] = quaternion
        
        num_connections = len(self.connection_indices)
        connection_data = np.zeros((T, num_connections, 3))
        
        for t in range(T):
            for c_idx, (a_idx, b_idx) in enumerate(self.connection_indices):
                point_a = trunk_data[t, a_idx, :2]
                point_b = trunk_data[t, b_idx, :2]
                
                d = np.linalg.norm(point_b - point_a)
                
                dx = point_b[0] - point_a[0]
                dy = point_b[1] - point_a[1]
                
                if dy == 0:
                    theta = np.pi/2 if dx > 0 else -np.pi/2
                else:
                    theta = math.atan(dx / dy)
                    if dy < 0:
                        theta += np.pi
                
                connection_data[t, c_idx, 0] = d 
                connection_data[t, c_idx, 1] = theta * 180
                connection_data[t, c_idx, 2] = t_vals[t]
        
        bone_gaussian_params = np.zeros((num_connections, K, 10))
        
        for c in range(num_connections):
            connection_trajectory = connection_data[:, c, :]
            
            gmm = GaussianMixture(
                n_components=K,
                covariance_type='full',
                init_params='kmeans',
                random_state=42
            )

            gmm.fit(connection_trajectory)
            
            for k in range(K):
                mean = gmm.means_[k]
                bone_gaussian_params[c, k, :3] = mean
                
                covariance = gmm.covariances_[k]
                
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                
                lambdas = np.sqrt(np.abs(eigenvalues))
                bone_gaussian_params[c, k, 3:6] = lambdas
                
                rotation_matrix = eigenvectors
                
                quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
                bone_gaussian_params[c, k, 6:] = quaternion
        
        used_bones = set()
        
        max_possible_size = J + num_connections
        interleaved_params = []
        
        for j in range(J):
            interleaved_params.append(joint_gaussian_params[j])
            
            if j in self.bone_to_joint_map:
                for bone_idx in self.bone_to_joint_map[j]:
                    if bone_idx not in used_bones:
                        interleaved_params.append(bone_gaussian_params[bone_idx])
                        used_bones.add(bone_idx)
        
        for c in range(num_connections):
            if c not in used_bones:
                interleaved_params.append(bone_gaussian_params[c])
        
        interleaved_params = np.array(interleaved_params)
        
        return torch.from_numpy(interleaved_params).float()
   
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