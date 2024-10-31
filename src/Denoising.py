from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from CustomImageDataset import CustomImageDataset

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Resize((256, 256))  # 调整图片大小
])

dataset_origin = CustomImageDataset(img_dir='../Data/Train/origin', transform=transform)
dataset_noise = CustomImageDataset(img_dir='../Data/Train/noise', transform=transform)
dataloader_origin = DataLoader(dataset_origin, batch_size=4, shuffle=True)
dataloader_noise = DataLoader(dataset_noise, batch_size=4, shuffle=True)

