import os

import numpy as np


import torch
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import random

from randaugment import RandAugment

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

class MTLDataset(torch.utils.data.Dataset):
    def __init__(self, root, images, labels, transform, target_transform=None):
        self.images = images
        self.labels = labels
        assert all([os.path.exists(os.path.join(root, img)) for img in self.images])

        self.root = root

        assert len(self.images) == len(self.labels)
        
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if len(self.labels.shape) == 3:
            target = self.labels[index, :, :]
        else:
            target = self.labels[index, :]

        path = self.images[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.labels)
    

def get_Dataset(args):

    if args.dataset == 'coco':
        images_train = np.load('coco/coco_train2014_images.npy')
        labels_train = np.load('coco/coco_train2014_labels.npy')

        images_val = np.load('coco/coco_val2014_images.npy')
        labels_val = np.load('coco/coco_val2014_labels.npy')

        if 'ecaresnet50t' in args.model_name:
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif 'vit_base_patch32_224' in args.model_name:
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            raise ValueError
        
        train_dataset = MTLDataset(root=os.path.join(args.data_dir, 'train2014'), images=images_train, labels=labels_train, transform=
                                transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        CutoutPIL(cutout_factor=0.5),
                                        # RandAugment(),
                                        # transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]), target_transform=None)
        
        val_dataset = MTLDataset(root=os.path.join(args.data_dir, 'val2014'), images=images_val, labels=labels_val, transform=
                                        transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        normalize, # no need, toTensor does normalization
                                    ]), target_transform=None)

    else:
        raise ValueError

    return train_dataset, val_dataset