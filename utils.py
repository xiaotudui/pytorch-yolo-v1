def calculate_iou(bbox1, bbox2):
    bbox1, bbox2 = bbox1.cpu().detach().numpy().tolist(), bbox2.cpu().detach().numpy().tolist()
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    # 计算两个框的面积
    area1 = w1 * h1
    area2 = w2 * h2

    # 计算两个框的左上角和右下角坐标
    left1, top1 = x1 - w1 / 2, y1 - h1 / 2
    right1, bottom1 = x1 + w1 / 2, y1 + h1 / 2
    left2, top2 = x2 - w2 / 2, y2 - h2 / 2
    right2, bottom2 = x2 + w2 / 2, y2 + h2 / 2

    # 计算交集区域的坐标
    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(bottom1, bottom2)

    # 计算交集面积
    if left < right and top < bottom:
        intersection = (right - left) * (bottom - top)
    else:
        intersection = 0

    # 计算并集面积
    union = area1 + area2 - intersection

    # 计算IOU
    iou = intersection / union if union > 0 else 0

    return iou