import cv2
import matplotlib.pyplot as plt
import os


def visualize_images(noise_img_path, denoise_img_path, index):
    noise_img = cv2.imread(noise_img_path)
    denoise_img = cv2.imread(denoise_img_path)

    # Convert BGR to RGB for displaying with matplotlib
    noise_img_rgb = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
    denoise_img_rgb = cv2.cvtColor(denoise_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"Noisy Image {index}")
    plt.imshow(noise_img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Denoised Image {index}")
    plt.imshow(denoise_img_rgb)
    plt.axis('off')

    plt.show()


# Example usage
noise_dir = '../Data/Test/noise'
denoise_dir = '../Data/Test/denoise'
noise_files = sorted(os.listdir(noise_dir))
denoise_files = sorted(os.listdir(denoise_dir))

for i, (noise_file, denoise_file) in enumerate(zip(noise_files, denoise_files)):
    noise_img_path = os.path.join(noise_dir, noise_file)
    denoise_img_path = os.path.join(denoise_dir, denoise_file)
    visualize_images(noise_img_path, denoise_img_path, i)
