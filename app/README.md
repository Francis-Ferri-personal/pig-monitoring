# 1. Create dedicated conda environment
conda create --prefix ./.venv-demo python=3.12 -y
conda activate ./.venv-demo

# 2. Install PyTorch for MMPose (CUDA 12.4)
<!-- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -->
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install OpenMMLab tools
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0" # Note: wheel building takes time
mim install "mmdet>=3.1.0"

# 4. Install MMPose from source
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
cd ..