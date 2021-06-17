# SIGIR 2021: Deconfounded Video Moment Retrieval with Causal Intervention

## Introduction

This is the repository for our SIGIR 2021 paper: [Deconfounded Video Moment Retrieval with Causal Intervention](https://arxiv.org/pdf/2106.01534.pdf). 

## Installation
The installation for this repository is easy. Please refer to [INSTALL.md](INSTALL.md).

## Dataset
Please refer to [DATASET.md](DATASET.md) to prepare datasets.

## Quick Start
We provide scripts for simplifying training and inference. Please refer to [scripts/train.sh](scripts/train.sh), [scripts/eval.sh](scripts/eval.sh).

For example, if you want to train TACoS dataset, just modifying [scripts/train.sh](scripts/train.sh) as follows:

```bash
# find all configs in configs/
model=2dtan_128x128_pool_k5l8_tacos
# set your gpu id
gpus=0,1,2,3
# number of gpus
gpun=4
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi 2dtan task on the same machine
master_addr=127.0.0.1
master_port=29501
...
```

Another example, if you want to evaluate on ActivityNet dataset, just modifying [scripts/eval.sh](scripts/eval.sh) as follows:

```bash
# find all configs in configs/
config_file=configs/2dtan_64x64_pool_k9l4_activitynet.yaml
# the dir of the saved weight
weight_dir=outputs/2dtan_64x64_pool_k9l4_activitynet
# select weight to evaluate
weight_file=model_1e.pth
# test batch size
batch_size=32
# set your gpu id
gpus=0,1,2,3
# number of gpus
gpun=4
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi 2dtan task on the same machine
master_addr=127.0.0.2
master_port=29502
...
```

## Support
Please feel free to contact me: hfutyangxun@gmail.com if you need my help.

## Acknowledgements
Please remember to cite our paper if you use our codes or features:
```
@InProceedings{yang2021deconfounded,
  title={Deconfounded Video Moment Retrieval with Causal Intervention},
  author={Yang, Xun and Feng, Fuli and Ji, Wei and Wang, Meng and Chua, Tat-Seng},
  booktitle={SIGIR},
  year={2021}
}
```

