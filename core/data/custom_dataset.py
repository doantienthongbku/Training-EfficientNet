import cv2
from pathlib import Path
from torch.utils.data import Dataset

# Dataset get from: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

class DogCat(Dataset):
    def __init__(self, image_dir, image_pattern='*.jpg', transforms=None):
        self.image_paths = sorted(list(Path(image_dir).glob(image_pattern)), key=lambda x: int(str(x).split('.')[-2]))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if 'cat' in image_path.stem:
            label = 0
        elif 'dog' in image_path.stem:
            label = 1
        else:
            label = -1

        return image, label