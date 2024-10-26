import os

import torch
from PIL.Image import Image
from torch.utils.data import Dataset

import config


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
        # 获得图片
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 从COCO数据集读取位置信息
        xymhc =[]
        with open(os.path.join(self.label_folder, img_name.split(".")[0] + ".txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                if line == "\n":
                    continue
                object_info = line.strip().split(" ")
                xymhc.append((float(i) for i in object_info))
        # 将COCO位置信息，转换为YOLOv1需要的数据形式
        label = torch.zeros(config.S, config.S, 5 * config.B + config.C)
        for x, y, w, h, c in xymhc:
            pass
        return img

