import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from CustomImageDataset import CustomImageDataset
from src.Model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Resize((256, 256))  # 调整图片大小
])

dataset_origin = CustomImageDataset(img_dir='../Data/Train/origin', transform=transform)
dataset_noise = CustomImageDataset(img_dir='../Data/Train/noise', transform=transform)
dataloader_origin = DataLoader(dataset_origin, batch_size=4, shuffle=True)
dataloader_noise = DataLoader(dataset_noise, batch_size=4, shuffle=True)

training = Model()

writer = SummaryWriter(log_dir='../logs')

step = 0
for data in dataloader_origin:
    img = data
    output = training(img)
    print(img.shape)
    print(output.shape)
    writer.add_images('Original Image', img, step)
    output = torch.reshape(output, (-1, 3, 256, 256))
    writer.add_images('Output Image', output, step)
    step += 1

writer.close()
