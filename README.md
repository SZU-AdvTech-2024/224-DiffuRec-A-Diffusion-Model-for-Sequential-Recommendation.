## Introduction
The report presents a reproducibility of the article titled "DiffuRec: A Diffusion Model for Sequential Recommendation".

This project refers to the source code of DiffuRec（https://github.com/WHUIR/DiffuRec) as the basic framework, and this project modifies and improves it to make it work properly, and to a certain extent.

This project is inspired by the EDiffuRec framework (EDiffRec is not open source), which suggests that the noise distribution can be replaced with other effective distributions. On this basis, this report attempts to use Weibull distribution instead of Gaussian distribution in DiffuRec for comparative experiments and analysis.

This project is inspired by the EDiffuRec framework (EDiffRec is not open source), which suggests that the noise distribution can be replaced with other effective distributions. 

The attempt to replace the distribution is:

1、The distribution learned by the approximator remains unchanged, and the noise distribution is replaced with a Weibull distribution in the forward diffusion process;

2、The distributions learned by diffusion forward process and reverse process are all replaced with Weibull distribution.

Only the second point was discussed in the EDiffuRec article, whereas the first point is based on my personal attempt.

## Requirements
- Python 3.6.9
- PyTorch 1.10.1
- CUDA 11.1 

## Usage

1. The code defaults to replacing Gaussian distribution with Weibull distribution.

2. You can run the below command for model training and evaluation.
```
   python main.py --dataset amazon_beauty
   python main.py --dataset toys
   python main.py --dataset steam
   python main.py --dataset ml-1m
```

## Dataset
The dataset download link is https://github.com/WHUIR/DiffuRec/tree/main/datasets

The datasets folder and src folder are at the same level of directory.

```
Project
   --datasets
   --src
```