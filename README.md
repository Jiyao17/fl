# fl
General multi-model test code for federated learning.

# Prerequisites
Sept 29 2021

Python 3.6.9

Options selected on pytorch official website:

Pytorch1.9.1 Linux Pip Python CUDA10.2 

Command used:

pip install --upgrade pip

pip3 install torch torchvision torchaudio torchtext

Device allowed:

CUDA, cpu

# Known Issues
When the dataset is not downloaded, do not run multiple simulation,
i.e., do not set configs.simulation.simulation_num > 1.

Set configs.simulation.simulation_num = 1 and let it download the dataset. Then restart.

# Usage
Add new task:

Inherite Task class in task.py, and modify UniTask. Please refer to existing tasks.

# TODO

Add support for:

batch configurations

splitting dataset in the non-IID manner
