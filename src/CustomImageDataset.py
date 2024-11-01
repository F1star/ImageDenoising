import cv2
import os
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, train_dir, transform=None):
        self.train_dir = train_dir
        self.origin_paths = [os.path.join(train_dir, 'origin', img_name) for img_name in
                             os.listdir(os.path.join(train_dir, 'origin'))]
        self.noise_paths = [os.path.join(train_dir, 'noise', img_name) for img_name in
                            os.listdir(os.path.join(train_dir, 'noise'))]
        self.transform = transform
        self.origin_paths.sort()
        self.noise_paths.sort()

    def __len__(self):
        return len(self.origin_paths)

    def __getitem__(self, idx):
        origin_img = cv2.imread(self.origin_paths[idx])
        noise_img = cv2.imread(self.noise_paths[idx])

        if self.transform:
            origin_img = self.transform(origin_img)
            noise_img = self.transform(noise_img)

        return noise_img, origin_img
