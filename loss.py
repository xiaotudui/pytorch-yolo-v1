import torch
from torch import nn

import config


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, preds, labels):
        batch_size = labels.size(0)
        for i in range(batch_size):
            for row in range(config.S):
                for col in range(config.S):
                    if labels[i, row, col, 4] == 1:
                        pred_bbox1 = preds[i, row, col, 0:4]
                        pred_bbox2 = preds[i, row, col, 5:8]
                        label_bbox = labels[i, row, col, 0:4]




if __name__ == '__main__':
    loss = YOLOLoss()
    preds = torch.rand(1, config.S, config.S, 2 * config.B + config.C)
    labels = torch.ones(1, config.S, config.S, 2 * config.B + config.C)
    loss(preds, labels)
