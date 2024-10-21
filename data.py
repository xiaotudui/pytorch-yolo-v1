from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torchvision.transforms as T

from config import Grid_S, IMAGE_SIZE


class YOLODataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.dataset = VOCDetection(root,
                                    year="2007",
                                    image_set = "train" if train else "test",
                                    download = True,
                                    transform = T.Compose([
                                        T.ToTensor(),
                                        T.Resize((IMAGE_SIZE[0], IMAGE_SIZE[1]))
                                    ]))

    def __getitem__(self, index):
        img, target = self.dataset[index]
        h, w = img.size()
        grid_count_x = w / Grid_S
        grid_count_y = h / Grid_S


