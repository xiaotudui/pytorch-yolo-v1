import torch
from torch import nn

import config
from utils import calculate_iou


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, preds, labels):
        batch_size = labels.size(0)

        loss_xy = 0

        for i in range(batch_size):
            for row in range(config.S):
                for col in range(config.S):
                    # 如果当前grid存在检测目标
                    if labels[i, row, col, 4] == 1:
                        pred_bbox1 = preds[i, row, col, 0:4]
                        pred_bbox2 = preds[i, row, col, 5:9]
                        label_bbox = labels[i, row, col, 0:4]
                        iou1 = calculate_iou(pred_bbox1, label_bbox)
                        iou2 = calculate_iou(pred_bbox2, label_bbox)

                        if iou1 > iou2:
                            return loss_xy






if __name__ == '__main__':
    loss = YOLOLoss()
    preds = torch.rand(1, config.S, config.S, 2 * config.B + config.C)
    labels = torch.ones(1, config.S, config.S, 2 * config.B + config.C)
    loss(preds, labels)
