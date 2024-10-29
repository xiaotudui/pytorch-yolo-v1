import argparse
import torch
from model import YOLOv1
from yolov1_dataset import YOLOv1Dataset

parser = argparse.ArgumentParser(description='Pytorch yolo v1 trainer')
parser.add_argument("--cfg", "-c", default="config/yolov1.yaml", help="path to config file")
parser.add_argument("--train", "-t", default="", help="path to train dataset")
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1().to(device)

    train_dataset = YOLOv1Dataset(img_folder="data/VOCdevkit/VOC2007/JPEGImages",
                                  label_folder="data/VOCdevkit/VOC2007/YOLOAnnotations")

    val_dataset = YOLOv1Dataset(img_folder="data/VOCdevkit/VOC2007/JPEGImages",
                                label_folder="data/VOCdevkit/VOC2007/YOLOAnnotations")

    print(model)
