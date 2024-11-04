import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from CustomImageDataset import CustomImageDataset
from Model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为Tensor，值范围在[0, 1]
])

dataset = CustomImageDataset(train_dir='../Data/Train', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # 将批量大小调整为16

model = Model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 将学习率降低到0.0001
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

num_epochs = 1000
early_stop_threshold = 0.0003  # 定义早停的损失阈值
early_stop_counter = 0
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data_noise, data_origin in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        data_noise, data_origin = data_noise.to(device), data_origin.to(device)
        optimizer.zero_grad()

        outputs = model(data_noise)
        loss = criterion(outputs, data_origin)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.6f}')

    # 学习率调度器步进
    scheduler.step(average_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Current learning rate: {current_lr:.6e}')

    # 保存最好的模型
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), '../Pth/Best_Checkpoint.pth')
        print(f'Best model saved at epoch {epoch + 1} with loss {best_loss:.6f}')

    # 早停策略
    if average_loss < early_stop_threshold:
        print(f"Loss has reached below {early_stop_threshold} at epoch {epoch + 1}. Stopping training.")
        break

# 保存最终模型
torch.save(model.state_dict(), '../Pth/Final_Checkpoint.pth')
