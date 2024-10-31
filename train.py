import argparse
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

import config
from loss import YOLOLoss
from model import YOLOv1
from yolov1_dataset import YOLOv1Dataset

parser = argparse.ArgumentParser(description='Pytorch yolo v1 by tudui')
args = parser.parse_args()

"""
用于训练
"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1().to(device)

    train_dataset = YOLOv1Dataset(img_folder="data/VOCdevkit/VOC2007/JPEGImages",
                                  label_folder="data/VOCdevkit/VOC2007/YOLOAnnotations",
                                  transform=Compose([
                                      Resize((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])),
                                      ToTensor()
                                  ]))

    val_dataset = YOLOv1Dataset(img_folder="data/VOCdevkit/VOC2007/JPEGImages",
                                label_folder="data/VOCdevkit/VOC2007/YOLOAnnotations",
                                transform=Compose([
                                    Resize((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])),
                                    ToTensor()
                                ]))

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=8, shuffle=False)

    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=8, shuffle=False)

    optimizer = SGD(model.parameters(), lr=0.01)

    best_loss = None
    for epoch in range(100):
        model.train()
        train_loss = 0
        for data in train_dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_fn = YOLOLoss()
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # print(loss.item())
        train_loss /= len(train_dataset)
        print("Epoch {} : Loss {}".format(epoch, train_loss))

        if best_loss is None:
            best_loss = train_loss
            torch.save(model.state_dict(), "weight/best.pth")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "weight/best.pth")
