import os
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

__all__ = ['CustomDataset']

class CustomDataset(Dataset):
    # Loads images from directory
    def __init__(self, image_folder, shape=(512, 512)):
        self.image_folder = image_folder
        self.shape = shape
        self.transform = transforms.Compose([
            transforms.Resize(size=(self.shape[0], self.shape[1]), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.image_list = glob(os.path.join(image_folder, '*.jpg'))

    def __getitem__(self, item):
        image = Image.open(self.image_list[item]).convert('RGB') # (C, H, W)
        image = self.transform(image)
        assert image.size(1) == self.shape[0] and image.size(2) == self.shape[1]
        return image

    def __len__(self):
        return len(self.image_list)
