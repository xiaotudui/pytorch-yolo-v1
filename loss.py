import torch
from torch import nn
import config
from utils import calculate_iou


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, preds, labels):
        batch_size = labels.size(0)

        loss_xy = 0.0
        loss_wh = 0.0
        loss_conf = 0.0
        loss_no_obj = 0.0
        loss_class = 0.0

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
                            loss_xy += 5 * torch.sum((labels[i, row, col, 0:2] - preds[i, row, col, 0:2]) ** 2)
                            loss_wh += 5 * torch.sum((torch.sqrt(labels[i, row, col, 2:4]) - torch.sqrt(preds[i, row, col, 2:4])) ** 2)
                            # todo: loss obj
                            loss_conf += (labels[i, row, col, 4] - preds[i, row, col, 4]) ** 2
                            # 另一个预测框
                            loss_no_obj += 0.5 * ((0 - preds[i, row, col, 9]) ** 2)
                        else:
                            loss_xy += 5 * torch.sum((labels[i, row, col, 5:7] - preds[i, row, col, 5:7]) ** 2)
                            loss_wh += 5 * torch.sum((torch.sqrt(labels[i, row, col, 7:9]) - torch.sqrt(preds[i, row, col, 7:9])) ** 2)
                            # todo: loss obj
                            loss_conf += (labels[i, row, col, 9] - preds[i, row, col, 9]) ** 2
                            loss_no_obj += 0.5 * ((0 - preds[i, row, col, 4]) ** 2)

                        loss_class += torch.sum((labels[i, row, col, 10:] - preds[i, row, col, 10:]) ** 2)

                    else:
                        loss_no_obj += 0.5 * ((0 - preds[i, row, col, 4]) ** 2 + (0 - preds[i, row, col, 9]) ** 2)

        loss = loss_xy + loss_wh + loss_conf + loss_no_obj + loss_class
        return loss / batch_size


if __name__ == '__main__':
    loss = YOLOLoss()
    preds = torch.rand(1, config.S, config.S, 2 * config.B + config.C)
    labels = torch.ones(1, config.S, config.S, 2 * config.B + config.C)
    print(loss(preds, labels))
