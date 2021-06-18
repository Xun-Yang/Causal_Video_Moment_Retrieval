## Installation

### Requirements:
- cudatoolkit 10.2
- python 3.7
- pytorch 1.5
- torchvision 0.6

### Installation
```bash
# create and activate a clean conda env
conda create -n DCM python=3.7
conda activate DCM

# install the right pip and dependencies for the fresh python
conda install ipython pip

# install some dependencies
pip install yacs h5py terminaltables tqdm pickle-mixin

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we use CUDA 10.2, others may also be okay
conda install pytorch torchvision cudatoolkit=10.2 torchtext -c pytorch

git clone https://github.com/Xun-Yang/Causal_Video_Moment_Retrieval
cd Causal_Video_Moment_Retrieval

# create the directory for storing logs and model checkpoints.
mkdir ./outputs

# create the dataset directories for Charades-STA
mkdir ./data/Charades-STA/charades_c3d
mkdir ./data/Charades-STA/charades_i3d
mkdir ./data/Charades-STA/charades_vgg
