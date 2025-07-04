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


## projects

* [trustworthy_dl](projects/trustworthy_dl/README.md): HEART'25 evaluation


## Citation
If you find the trustworthy_dl project useful, please cite our paper:

```{=latex}
@inproceedings{que2025trustworthy,
  title={Trustworthy Deep Learning Acceleration with Customizable Design Flow Automation},
  author={Que, Zhiqiang and Fan, Hongxiang and Figueiredo, Gabriel and Guo, Ce and Luk, Wayne and Yasudo, Ryota and Motomura, Masato},
  booktitle={Proceedings of the 15th International Symposium on Highly Efficient Accelerators and Reconfigurable Technologies},
  pages={1--13},
  year={2025}
}
```




