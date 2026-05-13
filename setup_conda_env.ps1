# Run this script from PowerShell after installing Anaconda/Miniconda and opening the FireDetection folder in VS Code.
# Make sure conda is available in the terminal before running.

conda create -n fire39 python=3.9 -y
conda activate fire39

python -m pip install --upgrade pip
python installPackages.py

# Install PyTorch CPU build for Python 3.9 using conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Optional: install PyQt5 tools if needed
python -m pip install PyQt5
python -m pip install pyqt5-tools==5.15.2.3.1

# Verify environment
python -c "import sys; print('Python:', sys.version); import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
