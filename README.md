# 🚀 FPGA-Accelerated Deep Learning for Real-Time Image Classification  

## 📌 Overview  
This project implements a **Convolutional Neural Network (CNN) on an Intel FPGA** for **real-time image classification**.  
The model was trained in **PyTorch**, converted to **ONNX**, optimized using **OpenVINO**, and deployed on **Intel DevCloud FPGA**.  

## 🛠️ Tech Stack  
✅ **PyTorch** (Model Training)  
✅ **ONNX** (Model Conversion)  
✅ **OpenVINO** (FPGA Optimization)  
✅ **Intel DevCloud FPGA** (Deployment & Testing)  

## 🔥 Key Achievements  
- **95.6% accuracy on CIFAR-10 dataset** after FPGA deployment.  
- **Successfully deployed optimized model on Intel DevCloud FPGA.**  

## 📌 Project Structure  
### FPGA-DeepLearning
 ```bash
models/                  # Contains PyTorch, ONNX, and OpenVINO models
    fpganet.py           # Trained PyTorch model
    fpganet.onnx         # ONNX format
    fpganet.xml          # OpenVINO IR format
    fpganet.bin          # OpenVINO binary model
    init.py              # Initiliazer for importing modules
training/                # Training and dataset scripts
    train.py             # PyTorch training script

conversion/              # Model conversion scripts
   optimize_fpga.py     # Converts ONNX to OpenVINO IR

inference/               # Inference and testing scripts
   run_fpga_inference.py # Runs inference on Intel FPGA
   test.py              # Tests on local images  
```

## 🚀 How to Run the Project  
### Clone the repository
```bash
git clone https://github.com/xk4r1x/FPGA-DeepLearning.git
```
```bash
cd FPGA-DeepLearning
```
```bash
pip install -r requirements.txt
python training/train.py
python inference/run_fpga_inference.py
```

### If you don't have a physical FPGA board
```bash
ssh <your_username>@devcloud.intel.com
scp -r fpganet_fpga <your_username>@devcloud.intel.com:~/fpga_models/
scp inference/test_image.jpg <your_username>@devcloud.intel.com:~/fpga_images/
source /opt/intel/openvino/setupvars.sh
python3 -c "import openvino.runtime; print('✅ OpenVINO is ready!')"
python3 inference/run_fpga_inference.py
```






