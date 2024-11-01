import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.Model import Model  # Import the model architecture

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the transformations (must match those used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Same normalization as training
])

# Load the model
model = Model().to(device)
model.load_state_dict(torch.load('../Pth/Checkpoints.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

def predict(image_path):
    # Load and transform the input image
    image = Image.open(image_path).convert('RGB')  # Ensure it's a 3-channel image
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(image)

    # Post-process the output (denormalize if necessary)
    output = output.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
    output = (output * 0.5) + 0.5  # Denormalize if you used Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    output = np.clip(output, 0, 1)  # Ensure values are in [0, 1]

    return output

# Example usage
predicted_image = predict('../Data/Test/noise/000000000139.jpg')  # Replace with your test image path

# Visualize the result
plt.imshow(np.transpose(predicted_image, (1, 2, 0)))  # Convert CHW to HWC for visualization
plt.title("Predicted Image")
plt.axis('off')
plt.show()
