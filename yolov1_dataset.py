import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

import config


# 加载YOLO格式的数据
class YOLOv1Dataset(Dataset):
    def __init__(self, img_folder, label_folder, transform=None):
        super().__init__()
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transform = transform
        self.img_names = os.listdir(self.img_folder)
        self.img_names.sort()

    def __len__(self):
        return len(self.img_names)

    def get_yolo_target(self, img_name):
        cxywh = []
        with open(os.path.join(self.label_folder, img_name.split(".")[0] + ".txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                if line == "\n":
                    continue
                object_info = line.strip().split(" ")
                cxywh.append([float(i) for i in object_info])
        return cxywh

    def get_yolo_img(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img_name, img

    def __getitem__(self, index):
        img_name, img = self.get_yolo_img(index)
        cxywh = self.get_yolo_target(img_name)

        label = torch.zeros(config.S, config.S, 5 * config.B + config.C)
        for c, x, y, w, h in cxywh:
            grid_x = int(x * config.S)
            grid_y = int(y * config.S)

            # 修改：计算相对于网格的坐标
            x_cell = x * config.S - grid_x  # 相对于网格的x坐标
            y_cell = y * config.S - grid_y  # 相对于网格的y坐标

            # 将预测值填入对应位置
            label[grid_y, grid_x, 0:5] = torch.tensor([x_cell, y_cell, w, h, 1])
            label[grid_y, grid_x, 5:10] = torch.tensor([x_cell, y_cell, w, h, 1])
            label[grid_y, grid_x, 10 + int(c)] = 1
        return img, label


if __name__ == '__main__':
    dataset = YOLOv1Dataset(img_folder="data/VOCdevkit/VOC2007/JPEGImages",
                            label_folder="data/VOCdevkit/VOC2007/YOLOAnnotations",
                            transform=Compose([ToTensor()]))
    data = dataset[1]
    print(data)
