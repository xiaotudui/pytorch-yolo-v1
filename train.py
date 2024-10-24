import argparse
import torch
from model import YOLOv1

parser = argparse.ArgumentParser(description='Pytorch yolo v1 trainer')
args = parser.parse_args()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1().to(device)