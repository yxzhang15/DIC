# DIC

## Introduction
This repository contains the source code for our work DIC. DIC is a deep-learning method designed for performing rigid partial-partial point cloud registration. 

## Configuration
python3.7,

PyTorch==1.8.1,

CUDA==11.1,

scipy,

tensorboardX,

h5py,

tqdm.

## Usage
Training employs the following commands. 
1. Train on Unkonwn Examples:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name=1
```

2. Train on Unkonwn Classes:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --unseen=true --exp_name=2
```

3. Train on Noise:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --gaussian_noise=true --clip=0.05 --exp_name=3 
```
