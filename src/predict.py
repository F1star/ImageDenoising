import torch
import cv2
from torchvision.transforms import transforms
from Model import Model
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义图像转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # 调整图片大小
    transforms.ToTensor(),  # 将图片转换为Tensor，值范围在[0, 1]
])

# 加载模型
model = Model().to(device)
model.load_state_dict(torch.load('../Pth/Checkpoints_epoch_50.pth', map_location=device))
model.eval()  # 设置为评估模式


# 定义预测函数
def predict_image(noise_image_path, output_path=None):
    # 读取并转换噪声图像
    noise_img = cv2.imread(noise_image_path)
    noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    noise_img = transform(noise_img).unsqueeze(0).to(device)  # 添加批次维度并转移到设备

    # 进行预测
    with torch.no_grad():
        output = model(noise_img)

    # 将输出转换回图像格式
    output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 去除批次维度并调整通道顺序
    output = (output * 255).astype('uint8')  # 转换为uint8格式，范围为[0, 255]

    # 保存或显示结果
    if output_path:
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # 转换回BGR格式以便保存
        cv2.imwrite(output_path, output_bgr)
        print(f"去噪图像已保存至 {output_path}")
    else:
        # 显示结果图像
        import matplotlib.pyplot as plt
        plt.imshow(output)
        plt.axis('off')
        plt.show()


# 主程序，供直接运行或导入调用
if __name__ == "__main__":
    # 示例：预测单张图像并保存结果
    input_noise_image = '../Data/Test/noise/000000000632.jpg'  # 替换为实际路径
    output_denoised_image = '../Data/Test/denoise/000000000632.jpg'  # 结果保存路径

    if not os.path.exists('../Data/Test/denoise'):
        os.makedirs('../Data/Test/denoise')

    predict_image(input_noise_image, output_denoised_image)
