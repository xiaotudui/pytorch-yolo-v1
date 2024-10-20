from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torchvision.transforms as T



class YOLODataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.dataset = VOCDetection(root,
                                    year="2007",
                                    image_set = "train" if train else "test",
                                    download = True,
                                    transform = T.Compose([
                                        T.ToTensor(),
                                        T.Resize(448)
                                    ]))

    def __getitem__(self, index):
        img, target = self.dataset[index]

        grid_size_x = img.size()[2]/7
        grid_size_y = img.size()[1]/7
        depth = 5 * 2 + 20

