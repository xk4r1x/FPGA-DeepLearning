import os
import cv2
import numpy as np
from openvino.runtime import Core

# ✅ Define the correct absolute path to the image
image_path = os.path.abspath("inference/dog.jpeg")

# ✅ Load and check if the image exists before resizing
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"❌ Error: The image file was not found at {image_path}. Check the file path.")

# ✅ Resize and preprocess the image
image = cv2.resize(image, (32, 32))  # Resize to match model input size
image = image.astype(np.float32) / 255.0  # Normalize pixel values
image = np.transpose(image, (2, 0, 1))  # Change to channel-first format
image = np.expand_dims(image, axis=0)  # Add batch dimension

# ✅ Load OpenVINO model
ie = Core()
model_path = "./fpganet_fpga/fpganet.xml"
compiled_model = ie.compile_model(model_path, "CPU")  # Use "FPGA" when real hardware is available

# ✅ Run inference
output_layer = compiled_model.output(0)
result = compiled_model([image])[output_layer]
predicted_class = np.argmax(result)

print(f"✅ FPGA Inference Complete! Predicted Class: {predicted_class}")
