import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import torchvision
from torchvision import models

from CustomImageDataset import CustomImageDataset
from Model import UNetModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]
])

dataset = CustomImageDataset(train_dir='../Data/Train', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # 批量大小调整为16

model = UNetModel().to(device)

# 使用 L1 损失函数
criterion = nn.L1Loss()

# 定义感知损失函数
vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False  # 冻结 VGG 网络的参数


def perceptual_loss(output, target):
    output_vgg = vgg(output)
    target_vgg = vgg(target)
    loss = nn.L1Loss()(output_vgg, target_vgg)
    return loss


# 使用 AdamW 优化器和较小的学习率
optimizer = optim.AdamW(model.parameters(), lr=0.00005)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

num_epochs = 200
early_stop_threshold = 0.0003  # 定义早停的损失阈值
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data_noise, data_origin in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        data_noise, data_origin = data_noise.to(device), data_origin.to(device)
        optimizer.zero_grad()

        outputs = model(data_noise)
        loss_pixel = criterion(outputs, data_origin)
        loss_percep = perceptual_loss(outputs, data_origin)
        loss = loss_pixel + 0.01 * loss_percep  # 调整感知损失的权重

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

    # 可视化模型输出
    model.eval()
    with torch.no_grad():
        sample_noise, sample_origin = next(iter(dataloader))
        sample_noise = sample_noise.to(device)
        sample_output = model(sample_noise)
        # 去归一化
        sample_output = sample_output * 0.5 + 0.5
        torchvision.utils.save_image(sample_output, f'../Outputs/output_epoch_{epoch + 1}.png')
    model.train()

# 保存最终模型
torch.save(model.state_dict(), '../Pth/Final_Checkpoint.pth')
