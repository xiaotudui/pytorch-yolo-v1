import torch
from torch import nn

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    model = YOLOv1()