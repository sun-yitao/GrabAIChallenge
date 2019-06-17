import os
from glob import glob
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset


__all__ = ['UnlabelledDataset', 'CsvDataset']

class UnlabelledDataset(Dataset):
    """Loads unlabelled images in directory
    Args:
        image_folder: path of directory to images
        transform: torchvision transforms
        shape: image size
    """
    def __init__(self, image_folder, transform, shape=(512, 512)):
        self.image_folder = image_folder
        self.transform = transform
        self.shape = shape
        self.image_list = glob(os.path.join(image_folder, '*.jpg'))

    def __getitem__(self, item):
        image = Image.open(self.image_list[item]).convert('RGB') # (C, H, W)
        image = self.transform(image)
        assert image.size(1) == self.shape[0] and image.size(2) == self.shape[1]
        return image

    def __len__(self):
        return len(self.image_list)


class CsvDataset(Dataset):
    """Loads unlabelled images in directory
    Args:
        image_folder: path of directory to images
        csv_path: path to csv containing image paths
        csv_headings, comma-separated heading names for image path and labels
        transform: torchvision transforms
        shape: image size
        relative_image_path: whether image paths in csv are relative to image folder or full path
    """
    def __init__(self, image_folder, csv_path, csv_headings, transform, shape=(512, 512), relative_image_path=True):
        self.image_folder = image_folder
        self.csv_path = csv_path
        self.shape = shape
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.csv_headings = csv_headings.split(',')
        self.relative_image_path = relative_image_path
        self.image_list = self.df[self.csv_headings[0]].values
        self.labels = self.df[self.csv_headings[1]].values
        self.samples = tuple(zip(self.image_list, self.labels))

    def __getitem__(self, item):
        if self.relative_image_path:
            img_path = os.path.join(self.image_folder, self.image_list[item])
            label = self.labels[item]
        else:
            img_path = self.image_list[item]
            label = self.labels[item]
        image = Image.open(img_path).convert('RGB') # (C, H, W)
        image = self.transform(image)
        assert image.size(1) == self.shape[0] and image.size(2) == self.shape[1]
        return image, label

    def __len__(self):
        return len(self.df)
