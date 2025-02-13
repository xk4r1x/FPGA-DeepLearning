import torch
import sys
import os

# Get the absolute path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Insert the project root at the BEGINNING of sys.path
sys.path.insert(0, project_root)  # This is the crucial change

from models.fpganet import FPGANet  # Import the CNN model

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FPGANet().to(device)
model.load_state_dict(torch.load("fpganet.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# Generate a dummy input tensor (same size as real input)
dummy_input = torch.randn(1, 3, 32, 32).to(device)

# Export model to ONNX format
torch.onnx.export(model, dummy_input, "fpganet.onnx")

print("✅ Model converted to ONNX format (fpganet.onnx)")
