# <p align="center">GAN---data-synthesizing </p>

<div align="center">
  <a href="https://hoangmn.com" target="_blank">Hoang&nbsp;Nguyen</a> &emsp; <b>&middot;</b> &emsp;
  <br> <br>
</div>
</p>
This repository contains sampled and simplified code of my research project on GAN in generating synthetic data.

## Understanding the code

This repo has the implementation of Vanilla GAN, [Conditional Tabular GAN](https://arxiv.org/abs/1907.00503) (ctgan), [Semi-Supervised GAN](https://arxiv.org/abs/1606.01583) (SGAN). This demo is used on Kaggle [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and a portion of Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) data. If you would like to use your own data, please adjust the preprocessing accordingly.

## Requirements

The experiments ran in Python 3.11. Please use the following command to install the requirements:
```shell script
pip install --upgrade pip
pip install -r requirements.txt
```