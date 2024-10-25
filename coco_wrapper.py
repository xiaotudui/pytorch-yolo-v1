import os

from PIL.Image import Image
from torch.utils.data import Dataset


class COCOWrapper(Dataset):
    def __init__(self, img_folder, label_folder, transform=None):
        super().__init__()
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transform = transform
        self.img_names = os.listdir(self.img_folder)
        self.img_names.sort()

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img