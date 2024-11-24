import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor

import config
from model import YOLOv1


"""
用于加载模型，并进行推理
"""

if __name__ == '__main__':
    model_path = "weight/finetune-kaggle.pth"
    input_img_path = "img2.jpg"
    # 加载模型
    model = YOLOv1()
    model.load_state_dict(torch.load(model_path))

    # 加载 图像
    input = Image.open(input_img_path).convert('RGB')
    transform = transforms.Compose([
        Resize((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])),
        ToTensor()
    ])

    input_tensor = transform(input)
    print(input_tensor.size())
    input_tensor.unsqueeze_(0)
    print(input_tensor.size())

    model.eval()
    with torch.no_grad():
        # 预测输出
        output = model(input_tensor)

    output = output.detach().cpu()[0]
    print(output.size())

    image = cv2.imread(input_img_path)
    image = cv2.resize(image, (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))

    for row in range(config.S):
        for col in range(config.S):
            preds = output[row, col]
            pred_classes = preds[10:]
            # plot first bbox in each cell
            pred_bbox1 = preds[:4]
            pred_conf1 = preds[4]
            if pred_conf1 > 0.1:
                x1 = int((col + pred_bbox1[0]) * (config.IMAGE_SIZE[0] / config.S))
                y1 = int((row + pred_bbox1[1]) * (config.IMAGE_SIZE[1] / config.S))
                w1 = int(pred_bbox1[2] * config.IMAGE_SIZE[0])
                h1 = int(pred_bbox1[3] * config.IMAGE_SIZE[1])
                cv2.rectangle(image, (x1 - w1 // 2, y1 - h1 // 2), (x1 + w1 // 2, y1 + h1 // 2), (0, 255, 0), 2)
                class_idx = torch.argmax(pred_classes).item()
                class_conf = torch.max(pred_classes).item()

                # 添加类别标签和置信度
                label = f"Class {class_idx}: {class_conf:.2f}"
                cv2.putText(image, label, (x1 - w1 // 2, y1 - h1 // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            pred_bbox2 = preds[5:9]
            pred_conf2 = preds[9]
            if pred_conf2 > 0.1:  # 设置置信度阈值
                x2 = int((col + pred_bbox2[0]) * (config.IMAGE_SIZE[0] / config.S))
                y2 = int((row + pred_bbox2[1]) * (config.IMAGE_SIZE[1] / config.S))
                w2 = int(pred_bbox2[2] * config.IMAGE_SIZE[0])
                h2 = int(pred_bbox2[3] * config.IMAGE_SIZE[1])

                cv2.rectangle(image, (x2 - w2 // 2, y2 - h2 // 2), (x2 + w2 // 2, y2 + h2 // 2), (255, 0, 0), 2)

                class_idx = torch.argmax(pred_classes).item()
                class_conf = torch.max(pred_classes).item()

                label = f"Class {class_idx}: {class_conf:.2f}"
                cv2.putText(image, label, (x2 - w2 // 2, y2 - h2 // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imwrite("detect.jpg", image)



