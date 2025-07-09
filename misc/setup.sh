# Install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

# Create conda environment
conda create -n stuart python=3.12
conda activate stuart
conda env config vars set PATH=/usr/local/cuda/bin:${PATH}
conda env config vars set LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
conda env config vars set THUNDERKITTENS_ROOT=...
conda env config vars set MEGAKERNELS_ROOT=...
conda env config vars set PYTHON_VERSION=3.12
conda env config vars set GPU=H100

# B200-specific torch (may be stable later)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# C++/Python binding
pip install pybind11

# Flash attention
pip install packaging ninja # required by flash attention
git clone https://github.com/Dao-AILab/flash-attention # use >=2.6.0
cd flash-attention
python setup.py install
