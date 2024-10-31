import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor

import config

"""
用于加载模型，并进行推理
"""

if __name__ == '__main__':
    model_path = "yolo.weights"
    input_img_path = "img1.jpg"
    # 加载模型
    model = torch.load(model_path)
    # 加载 图像
    input = Image.open(input_img_path).convert('RGB')
    transform = transforms.Compose([
        Resize((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])),
        ToTensor()
    ])
    input_tensor = transform(input)

    # 预测输出
    output = model(input_tensor)


