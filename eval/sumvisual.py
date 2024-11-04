import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from eval.SumImageData import SumImageData

# Define the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize images
    transforms.ToTensor(),  # Convert images to Tensor
])

# Initialize dataset
dataset = SumImageData(train_dir='../Data/Test', transform=transform)

# Initialize SummaryWriter
writer = SummaryWriter(log_dir='../logs')

# Constants
batch_size = 128
global_step = 0  # To keep track of the global step for TensorBoard logging

# Iterate over the dataset in batches of 32
for i in range(0, len(dataset), batch_size):
    noise_images = []
    denoise_images = []

    # Collect a batch of images
    for j in range(batch_size):
        try:
            noise, denoise = dataset[i + j]
            noise_images.append(noise)
            denoise_images.append(denoise)
        except IndexError:
            # If the dataset size is not perfectly divisible by batch_size, the last batch might be smaller
            break

            # Stack and make grid if there are images collected
    if noise_images and denoise_images:
        # Stack images to create a batch (convert list to tensor)
        noise_batch = torch.stack(noise_images)
        denoise_batch = torch.stack(denoise_images)

        # Use make_grid to create an image grid
        nrow_value = 32  # Adjust this value to control the number of images per row
        noise_grid = make_grid(noise_batch, nrow=nrow_value, padding=2)
        denoise_grid = make_grid(denoise_batch, nrow=nrow_value, padding=2)

        # Log the image grids
        writer.add_image("Denoise_Grid", denoise_grid, global_step)
        writer.add_image("Noise_Grid", noise_grid, global_step)

        # Increment the global step
        global_step += 1

    # Close the writer
writer.close()