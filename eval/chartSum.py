import cv2
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Initialize SummaryWriter
writer = SummaryWriter(log_dir='../logs')

# Directories for noise and denoise images
noise_dir = '../Data/Test/noise'
denoise_dir = '../Data/Test/denoise'

# Get sorted list of image files
noise_files = sorted(os.listdir(noise_dir))
denoise_files = sorted(os.listdir(denoise_dir))


def plot_histogram_image(img, title):
    colors = ('b', 'g', 'r')

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_title(title)
    ax.axis('off')

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)

    # Convert plot to image
    fig.canvas.draw()
    histogram_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    histogram_image = histogram_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    return histogram_image


def create_image_grid(images, grid_size=(32, 4)):
    rows, cols = grid_size
    h, w, c = images[0].shape
    grid_image = np.zeros((rows * h, cols * w, c), dtype=np.uint8)

    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        row, col = divmod(idx, cols)
        grid_image[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = img

    return grid_image


# Process images in batches of 128
batch_size = 128
for batch_start in range(0, len(noise_files), batch_size):
    noise_images = []
    denoise_images = []

    for i in range(batch_start, min(batch_start + batch_size, len(noise_files))):
        noise_img_path = os.path.join(noise_dir, noise_files[i])
        denoise_img_path = os.path.join(denoise_dir, denoise_files[i])

        # Read images using OpenCV
        noise_img = cv2.imread(noise_img_path)
        denoise_img = cv2.imread(denoise_img_path)

        # Plot histograms and get images
        noise_histogram_image = plot_histogram_image(noise_img, f"Noisy {i}")
        denoise_histogram_image = plot_histogram_image(denoise_img, f"Denoised {i}")

        noise_images.append(noise_histogram_image)
        denoise_images.append(denoise_histogram_image)

    # Create grid images
    noise_grid_image = create_image_grid(noise_images, grid_size=(4, 32))
    denoise_grid_image = create_image_grid(denoise_images, grid_size=(4, 32))

    # Log the grid images to TensorBoard
    batch_index = batch_start // batch_size
    writer.add_images(f"Noise_Histogram", noise_grid_image, batch_index, dataformats='HWC')
    writer.add_images(f"Denoise_Histogram", denoise_grid_image, batch_index, dataformats='HWC')

# Close the writer
writer.close()