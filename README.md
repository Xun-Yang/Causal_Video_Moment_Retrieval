# SIGIR 2021: Deconfounded Video Moment Retrieval with Causal Intervention

## Introduction

This is the repository for our SIGIR 2021 paper: [Deconfounded Video Moment Retrieval with Causal Intervention](https://dl.acm.org/doi/pdf/10.1145/3404835.3462823). 

## Installation
The installation for this repository is easy. Please refer to [INSTALL.md](INSTALL.md).

## Dataset
Please refer to [DATASET.md](DATASET.md) to prepare datasets.

## Quick Start
We provide scripts for simplifying training and inference. Please refer to scripts/train_xx.sh, scripts/eval_xx.sh.

For example, if you want to train our TCN+DCM method on the Charades-STA dataset, just modifying [scripts/train_charades_DCM.sh](scripts/train_charades_DCM.sh) to select the features (C3D/I3D/VGG), set your gpu id and set the number of training rounds with different random seeds (In our experiments, report the average performance of 10 runs on Charades-STA and DiDeMo, and 5 runs on ActivityNet-Captions, with different random seeds for network initialization). 

Start training: bash ./scripts/train_charades_DCM.sh


## Support
Please feel free to contact me: hfutyangxun@gmail.com if you need any help.

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
We appreciate the optimized 2D-Tan repository https://github.com/ChenJoya/2dtan. We use some codes from the optimized 2D-Tan repository to implement our work. Please also cite their paper if you use the codes.

