from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from pathlib import Path

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

def get_dataset(name):
    if name == 'lol':
        trainpath = Path.cwd()/'data/lol_dataset/our485/low'
        valpath = Path.cwd()/'data/lol_dataset/eval15/low'
        trainds = CustomDataset(trainpath)
        valds = CustomDataset(valpath)

        return trainds, valds