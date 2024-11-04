import torch
import cv2
from torchvision.transforms import transforms
from training.Model import Model
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
model.load_state_dict(torch.load('./Pth/Best_Checkpoint.pth', map_location=device))
model.eval()  # 设置为评估模式

# 定义预测函数
def predict_image(noise_image_path, output_path):
    # 读取并转换噪声图像
    noise_img = cv2.imread(noise_image_path)
    original_size = (noise_img.shape[1], noise_img.shape[0])  # 保存原始尺寸 (width, height)
    noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    noise_img = transform(noise_img).unsqueeze(0).to(device)  # 添加批次维度并转移到设备

    # 进行预测
    with torch.no_grad():
        output = model(noise_img)

    # 将输出转换回图像格式
    output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 去除批次维度并调整通道顺序
    output = (output * 255).astype('uint8')  # 转换为uint8格式，范围为[0, 255]

    # 调整输出图像大小回到原始尺寸
    output = cv2.resize(output, original_size)

    # 保存结果
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # 转换回BGR格式以便保存
    cv2.imwrite(output_path, output_bgr)
    print(f"去噪图像已保存至 {output_path}")

# 主程序，处理整个文件夹
if __name__ == "__main__":
    input_folder = './Data/Test/noise'  # 输入文件夹路径
    output_folder = './Data/Test/denoise'  # 输出文件夹路径

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 根据需要调整文件扩展名
            input_noise_image = os.path.join(input_folder, filename)
            output_denoised_image = os.path.join(output_folder, filename)
            predict_image(input_noise_image, output_denoised_image)
