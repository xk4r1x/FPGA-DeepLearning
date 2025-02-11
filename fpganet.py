import torch
import torch.nn as nn  # Importing PyTorch's neural network module

# Define a Convolutional Neural Network (CNN) optimized for FPGA deployment
class FPGANet(nn.Module):
    def __init__(self):
        """
        Initializes the CNN model with two convolutional layers,
        followed by fully connected layers for classification.
        """
        super(FPGANet, self).__init__()

        # First convolutional layer: 3 input channels (RGB), 16 filters, 3x3 kernel, padding=1 (keeps spatial size the same)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # Second convolutional layer: 16 input channels, 32 filters, 3x3 kernel, padding=1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fully connected layer (FC1): Takes the flattened output from the conv layers (32x8x8) and maps it to 128 neurons
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)

        # Fully connected layer (FC2): Maps the 128 features to 10 output classes (for CIFAR-10)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """
        Defines the forward pass for the CNN model.

        Input: 3-channel image tensor of shape (batch_size, 3, 32, 32)
        Output: Logits for 10 classes (before softmax)
        """
        x = torch.relu(self.conv1(x))  # Apply first convolution + ReLU activation
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # Downsample using 2x2 max pooling

        x = torch.relu(self.conv2(x))  # Apply second convolution + ReLU activation
        x = torch.max_pool2d(x, kernel_size=2, stride=2)  # Downsample again

        x = x.view(-1, 32 * 8 * 8)  # Flatten the tensor for the fully connected layers

        x = torch.relu(self.fc1(x))  # First fully connected layer with ReLU activation
        x = self.fc2(x)  # Final layer outputs raw class scores (logits)

        return x  # No softmax applied here since loss function will handle it
