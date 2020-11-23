from torch.utils import data
from PIL import Image
import pandas as pd 
import os

import torchvision.transforms as tvtf

class ImageFolderDataset(data.Dataset):
    def __init__(self, csv):
        self.df = pd.read_csv(csv) 
        self.files = self.df['filename']
        self.labels = self.df['label']
        self.transforms = tvtf.Compose([
            tvtf.Resize((224, 224)),
            tvtf.ToTensor(),
            tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img_path = self.files[index]
        label = self.labels[index]
        im = Image.open(img_path).convert('RGB')
        im = self.transforms(im)
        return im, label 

    def __len__(self):
        return len(self.labels)
