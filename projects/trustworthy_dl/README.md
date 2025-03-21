## Conda Environment

To create the conda environment:
```bash
bash conda.env.build
```

To activate:
```bash
conda activate metaml2
```

Check your cuda installation:
```bash
(metaml2) python3 cuda-test.py
```

## Execution

To search the optimal model of lenet on the mnist dataset with Bayesian OPT, run
```
bash run.sh
```

To train the optimal model of resnet on cifar10 dataset with Bayesian OPT, run
```
bash run_resnet.sh
```

To change the score function in the bayesian OPT search, check the link 420 in the file of train_bayesian_opt.py.


