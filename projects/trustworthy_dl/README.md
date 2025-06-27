## Bayesian Optimization with Bayesian Neural Networks for Image Classification

This script trains various quantized or standard models (e.g., LeNet, ResNet, VGG) on datasets like MNIST, CIFAR-10, or SVHN.
It applies model compression techniques such as pruning and dropout, and performs Bayesian optimization on hyperparameters
such as dropout rate, pruning rate, number of Bayesian layers, and scaling factors.

### Main Features:
- Dataset loading and preprocessing (MNIST, CIFAR-10, SVHN)
- Model selection (standard and quantized versions of LeNet, ResNet, VGG)
- Pruning and quantization support (via TensorFlow Model Optimization and QKeras)
- Support for Bayesian/MCDropout and Masksembles for uncertainty estimation
- Monte Carlo (MC) sampling-based predictions and expected calibration error (ECE) evaluation
- Hyperparameter optimization via Bayesian Optimization
- Result saving, including FLOPs, ECE, entropy, accuracy, and weighted score

### Conda Environment Setup
To create the MetaML conda environment:
```bash
bash conda.env.build
```

To activate:
```bash
conda activate metaml2
```

### Execution
To search the optimal model of LeNet on the MNIST dataset with Bayesian Optimization, run:
```bash
bash run_lenet.sh
```

To train the optimal model of ResNet on the CIFAR-10 dataset with Bayesian Optimization, run:
```bash
bash run_resnet.sh
```

### Parameterizing and Customizing `train_bayesian_opt.py`
You can fine-tune the behavior of the Bayesian optimization process by directly modifying or extending the arguments passed to `train_bayesian_opt.py`. Below are some commonly used parameters:

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset` | Dataset to use (`mnist`, `cifar10`, `svhn`) | `--dataset mnist` |
| `--model_name` | Model type (`lenet`, `resnet`, `vgg`, `vibnn`) | `--model_name resnet` |
| `--save_dir` | Directory to save experiment results | `--save_dir ./results/exp1` |
| `--is_quant` | Enable quantization (0 or 1) | `--is_quant 1` |
| `--num_epoch` | Number of training epochs | `--num_epoch 100` |
| `--dropout_type` | Dropout type: `mc` or `mask` | `--dropout_type mc` |
| `--w_list` | Score weights `[acc, ape, ece, flops]` | `--w_list 0.25,0.25,0.25,0.25` |

All available arguments can be viewed by inspecting the `argparse` section in the script.

### Changing the Score Function
To modify how the final model score is calculated during the Bayesian optimization loop, refer to **line 420** in `train_bayesian_opt.py`. This is where the weighted score is computed:
```python
final_score = float(accuracy / acc_base[args.model_name])   * w_base["acc"] \
            + float(ape      / ape_base[args.model_name])   * w_base["ape"] \
            - float(ece      / ece_base[args.model_name])   * w_base["ece"] \
            - float(flops    / flops_base[args.model_name]) * w_base["flops"]
```
You can customize this formula by changing the weights or the logic to better suit your optimization goals, such as prioritizing lower entropy or fewer FLOPs.

For example, if you want to heavily prioritize accuracy and ignore FLOPs:
```python
final_score = float(accuracy / acc_base[args.model_name]) * 0.7 \
            - float(ece / ece_base[args.model_name]) * 0.3
```
Make sure the weights (`w_list`) passed as arguments align with any changes in this logic.

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
