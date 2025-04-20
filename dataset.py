from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from pathlib import Path
import torch

class CustomDataset(Dataset):
    def __init__(self, image_path):
        self.files = list()
        for file in image_path.iterdir():
            self.files.append(file)
        self.transform = T.Compose([
            T.PILToTensor()
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        image = Image.open(self.files[index])
        image = self.transform(image)
        image = image/255.0
        return image

def pair_downsampler(img):
    #img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)

    output1 = torch.nn.functional.conv2d(img, filter1, stride=2, groups=c)
    output2 = torch.nn.functional.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2


def get_dataset(name):
    if name == 'lol':
        trainpath = Path.cwd()/'data/lol_dataset/our485/low'
        valpath = Path.cwd()/'data/lol_dataset/eval15/low'
        trainds = CustomDataset(trainpath)
        valds = CustomDataset(valpath)

        return trainds, valds