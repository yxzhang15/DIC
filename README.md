# Constructing Diverse Inlier Consistency for Partial Point Cloud Registration

## Introduction
This repository contains the source code for DIC. DIC is a deep-learning method designed for performing rigid partial-partial point cloud registration for objects. 

# Configuration
python3.7,
PyTorch==1.8.1,
CUDA==11.1,
scipy,
tensorboardX,
h5py,
tqdm.

# Usage

Train on Unkonwn Samples:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name=1
```

Train on Unkonwn Classes:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --unseen=true --exp_name=2
```

Train on Noise:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --gaussian_noise=true --clip=0.05 --exp_name=3 
```
