import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import *

class GaussianCNN(nn.Module):
    def __init__(self, num_joints=21, gmm_components=25, gaussian_params=10, num_classes=22):
        super(GaussianCNN, self).__init__()
        
        self.num_joints = num_joints
        self.gmm_components = gmm_components
        self.gaussian_params = gaussian_params
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(gaussian_params, 64, kernel_size=3, padding=1),
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
        
        h_out = num_joints // 2 // 2 // 2
        w_out = gmm_components // 2 // 2 // 2
        h_out = max(1, h_out)
        w_out = max(1, w_out)
        
        self.fc_input_size = 256 * h_out * w_out
        print(f"Linear layer input size: {self.fc_input_size}")
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
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
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv1(x)
        
        x = self.conv2(x)
        
        x = self.conv3(x)
        
        global_feat = self.global_pool(x)
        global_feat = global_feat.view(batch_size, -1)
        global_logits = self.global_fc(global_feat)
        
        x = x.reshape(batch_size, -1)
        
        main_logits = self.classifier(x)
        
        logits = main_logits + 0.2 * global_logits
        
        return logits