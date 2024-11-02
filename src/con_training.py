import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from CustomImageDataset import CustomImageDataset
from src.Model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为Tensor，值范围在[0, 1]
    # 移除Normalize步骤
])

dataset = CustomImageDataset(train_dir='../Data/Train', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = Model().to(device)
model.load_state_dict(torch.load('../Pth/Checkpoints.pth', map_location=device))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data_noise, data_origin in tqdm(dataloader, desc=f'Epoch {epoch + 1}', total=len(dataloader)):
        data_noise, data_origin = data_noise.to(device), data_origin.to(device)
        optimizer.zero_grad()

        outputs = model(data_noise)
        loss = criterion(outputs, data_origin)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'../Pth/Checkpoints_epoch_{epoch + 1}.pth')
        print(f'Model saved at epoch {epoch + 1}')

torch.save(model.state_dict(), '../Pth/Checkpoints.pth')
