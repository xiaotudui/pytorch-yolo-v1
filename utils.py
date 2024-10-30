def calculate_iou(bbox1, bbox2):
    bbox1, bboxes2 = bbox1.cpu().detach().numpy().tolist(), bbox2.cpu().detach().numpy().tolist()
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    area1 = w1 * h1
    area2 = w2 * h2
    return 0

