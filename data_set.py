import os
from torchvision import transforms
from torch.utils.data import Dataset
import imageio.v2 as io 
import matplotlib.pyplot as plt
from IPython.display import display, Image
from PIL import Image
import torch

#from typing import Any


class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        #self.classes=sorted(os.listdir(root_dir))
        self.classes= [class_name for class_name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, class_name))]

        self.data = []
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir,class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.data.append((image_path,i))
        
            

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label




def custom_collate(batch):
    # This collate function handles variable-sized images in the batch
    images, labels = zip(*batch)
    images = [img.unsqueeze(0) for img in images]  # Add batch dimension
    images = torch.cat(images, dim=0)
    labels = torch.tensor(labels)
    return images, labels

import os
from torchvison.io import read_image
import pandas as pd
from torch.utils.data import Dataset


class FashionMINST(Dataset):

    def __init__(self, annotation_file, image_dir, image_transform=None, target_transform=None):

        self.image_labels = pd.read_csv(annotation_file)
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_labels.iloc[idx,0])
        image = read_image(image_path)
        label = self.image_labels.iloc[idx,1]

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label 


        

