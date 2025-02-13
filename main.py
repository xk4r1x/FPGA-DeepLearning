import os

print("🚀 Training Model...")
os.system("python training/train.py")

print("🔄 Converting Model to ONNX...")
os.system("python conversion/convert_to_onnx.py")

print("⚡ Optimizing Model for FPGA...")
os.system("python conversion/optimize_fpga.py")

print("✅ Model is ready for FPGA deployment!")
