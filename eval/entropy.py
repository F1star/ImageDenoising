import cv2
from skimage.measure import shannon_entropy
import os


def calculate_entropy(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    entropy = shannon_entropy(image)
    return entropy


noise_dir = '../Data/Test/noise'
denoise_dir = '../Data/Test/denoise'
noise_files = sorted(os.listdir(noise_dir))
denoise_files = sorted(os.listdir(denoise_dir))

# Example usage
for i, (noise_file, denoise_file) in enumerate(zip(noise_files, denoise_files)):
    noise_img_path = os.path.join(noise_dir, noise_file)
    denoise_img_path = os.path.join(denoise_dir, denoise_file)

    noisy_entropy = calculate_entropy(noise_img_path)
    denoised_entropy = calculate_entropy(denoise_img_path)

    print(f"Image {i}:")
    print(f"  Noisy Image Entropy: {noisy_entropy}")
    print(f"  Denoised Image Entropy: {denoised_entropy}")
