import csv
import os
import tarfile
from urllib.parse import urlparse
from urllib.request import urlretrieve
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

object_categories = ["waste_bag","metal","shoe","plastic","bottle","carton","lile","galss"]



def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
    return data


def read_object_labels(root, dataset, phase):
    path_labels = os.path.join(root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + phase + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(8):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = torch.from_numpy((np.asarray(row[1:num_categories + 1])).astype(np.float32))
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


# def find_images_classification(root, dataset, phase):
#     path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
#     images = []
#     file = os.path.join(path_labels, phase + '.txt')
#     with open(file, 'r') as f:
#         for line in f:
#             images.append(line)
#     return images





class Waste(Dataset):
    def __init__(self, root, phase, transform=None):
        self.root = os.path.abspath(root)
        self.path_devkit = os.path.join(self.root, 'VOCdevkit')
        self.path_images = os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.phase = phase
        self.transform = transform
       

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'VOC2007')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + phase + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2007', self.phase)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)
        print('[dataset] VOC 2007 classification phase={} number of classes={}  number of images={}'.format(phase, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        filename, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, filename + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        data = {'image':img, 'name': filename, 'target': target}
        return data
        # image = {'image': img, 'name': filename}
        # return image, target
        # return (img, filename), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)



