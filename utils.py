import config


# 从VOC数据集标签中获得对应的位置坐标，并对其进行缩放到设置输入的大小
def get_voc_bounding_boxes(label):
    size = label["annotation"]["size"]
    width = int(size["width"])
    height = int(size["height"])
    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    obj_name_boxes = []
    objects = label["annotation"]["object"]
    for obj in objects:
        box = obj["bndbox"]
        coords = (
            int(int(box["xmin"]) * x_scale),
            int(int(box["ymin"]) * y_scale),
            int(int(box["xmax"]) * x_scale),
            int(int(box["ymax"]) * y_scale),
        )
        obj_name = obj["name"]
        obj_name_boxes.append((obj_name, coords))
    return obj_name_boxes
