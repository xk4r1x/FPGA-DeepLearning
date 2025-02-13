import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import os

# Get the absolute path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Insert the project root at the BEGINNING of sys.path
sys.path.insert(0, project_root)  # This is the crucial change

from models.fpganet import FPGANet  # Import the CNN model

# ✅ Load CIFAR-10 test dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# ✅ Get one CIFAR-10 test image
image, label = next(iter(testloader))  # Pick a random test image
image_np = image.numpy().squeeze().transpose(1, 2, 0)  # Convert tensor to NumPy image

# ✅ Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FPGANet().to(device)
model.load_state_dict(torch.load("fpganet.pth", map_location=device))
model.eval()

# ✅ Run inference on CIFAR-10 test image
image = image.to(device)
output = model(image)
predicted_class = torch.argmax(output).item()

print(f"✅ True Label: {label.item()} | Predicted Class: {predicted_class}")
