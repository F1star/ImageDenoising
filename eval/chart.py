import cv2
import matplotlib.pyplot as plt
import os


def plot_histograms(noise_img_path, denoise_img_path, index):
    noise_img = cv2.imread(noise_img_path)
    denoise_img = cv2.imread(denoise_img_path)

    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"Noisy Image Histogram {index}")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([noise_img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)

    plt.subplot(1, 2, 2)
    plt.title(f"Denoised Image Histogram {index}")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([denoise_img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)

    plt.show()

noise_dir = '../Data/Test/noise'
denoise_dir = '../Data/Test/denoise'
noise_files = sorted(os.listdir(noise_dir))
denoise_files = sorted(os.listdir(denoise_dir))

# Example usage
for i, (noise_file, denoise_file) in enumerate(zip(noise_files, denoise_files)):
    noise_img_path = os.path.join(noise_dir, noise_file)
    denoise_img_path = os.path.join(denoise_dir, denoise_file)
    plot_histograms(noise_img_path, denoise_img_path, i)
