import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import sys
import os

# Get the absolute path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Insert the project root at the BEGINNING of sys.path
sys.path.insert(0, project_root)  # This is the crucial change

from models.fpganet import FPGANet  # Import the CNN model


# Define data preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32 (CIFAR-10 size)
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1,1] range
])

# Load CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Check for GPU availability and move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FPGANet().to(device)

# Define loss function (CrossEntropyLoss for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Define optimizer (Adam optimizer for better convergence)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (10 epochs)
for epoch in range(20):
    running_loss = 0.0  # Track cumulative loss for the epoch
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available

        optimizer.zero_grad()  # Reset gradients to prevent accumulation
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss

    # Print loss after each epoch
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

# Save trained model to disk
torch.save(model.state_dict(), "fpganet.pth")
print("✅ Training Complete! Model saved as 'fpganet.pth'")
