import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
import config
import utils


class YOLODataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.dataset = VOCDetection(root,
                                    year="2007",
                                    image_set = "train" if train else "test",
                                    download = True,
                                    transform = T.Compose([
                                        T.ToTensor(),
                                        T.Resize((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
                                    ]))
        # 将所有的目标类型进行编码
        self.classes = config.VOC_CLASSES

    def __getitem__(self, index):
        img, target = self.dataset[index]
        c, h, w = img.size()
        grid_x = w / config.S
        grid_y = h / config.S
        depth = config.C + 5 * config.B
        ground_truth = torch.zeros(config.S, config.S, depth)

        for obj_name, obj_bbox in utils.get_voc_bounding_boxes(target):
            x1, y1, x2, y2 = obj_bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_col_index = int(center_x/grid_x)
            center_row_index = int(center_y/grid_y)
            # 在对应的cell，将 类别 one hot 填充进去
            class_index = self.classes.index(obj_name)
            class_one_hot = torch.zeros(config.C)
            class_one_hot[class_index] = 1
            ground_truth[center_row_index, center_col_index, config.B*5:] = class_one_hot
            # 将对应的位置坐标封装下
            bbox_truth = (
                (center_x - center_col_index * grid_x) / config.IMAGE_SIZE[0],
                (center_y - center_row_index * grid_y) / config.IMAGE_SIZE[1],
                (x2 - x1) / config.IMAGE_SIZE[0],
                (y2 - y1) / config.IMAGE_SIZE[1],
                1.0
            )
            ground_truth[center_row_index, center_col_index, : 5] = torch.tensor(bbox_truth)

        return ground_truth







        # 首先将目标类型转为对应的结果




if __name__ == '__main__':
    train_dataset = YOLODataset("./data", True)
    print(train_dataset[0])