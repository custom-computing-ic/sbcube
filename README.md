# a playground for sbcube developers

## requirements
1. Install `Anaconda` or `Miniconda` (recommended). To install Miniconda, follow the instructions [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers). This is a powerful package manager that simplifies the installation, updating, and management of software packages and their dependencies, allowing to create and manage isolated environments, which helps avoid conflicts between different versions of packages and dependencies.
2. Speed-up conda solver (not required but highly recommended) by installing [libnamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community):
   ```bash
   conda update -n base conda
   conda install -n base conda-libmamba-solver
   conda config --set solver libmamba
   ```
2. Install g++ (recommended version: 11) using your Linux distro manager:
    ```bash
    apt install g++
    ```
3. If using CUDA, ensure the NVIDIA driver is installed and `nvidia-smi` works.

## installation

The installation process sets up a conda environment with all the necessary tools and runtimes required to execute the experiments available in this repository. The following packages will be installed:

- Python 3.11
- PyTorch/LibTorch 2.41
- CUDA Toolkit 11.8
- Artisan Metaprogramming 1.0

To create the conda environment:
```bash
conda env create -f environment.yml
```

To activate:
```bash
conda activate sbcube
```

Check your cuda installation:
```bash
(sbcube) python3 cuda-test.py
```

## projects

* [cpp-torch/](cpp-torch/README.md): libtorch (C++) experiments using metaprogramming






