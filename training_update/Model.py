import torch
from torch import nn


class UNetModel(nn.Module):
    def __init__(self):
        super(UNetModel, self).__init__()
        # 编码器层
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # 解码器层
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 连接跳跃连接
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 连接跳跃连接
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 连接跳跃连接
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        # 编码器
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        # 瓶颈
        bn = self.bottleneck(p3)
        # 解码器
        up3 = self.up3(bn)
        merge3 = torch.cat([up3, d3], dim=1)
        dec3 = self.dec3(merge3)
        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, d2], dim=1)
        dec2 = self.dec2(merge2)
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, d1], dim=1)
        dec1 = self.dec1(merge1)
        out = self.final(dec1)
        out = self.activation(out)
        return out
