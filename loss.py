from torch import nn


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, output, target):
        output_w = output[..., 2]
