# P-02: Classifying flower images


## Introduction and Approach
The image classifier project is part of Udacity's Data Scientist Nanodegree. The goal is to build, train, and test a neural network to identify 102 different flower species. The neural network is implemented with PyTorch and uses pre-trained convolutional deep networks (e.g. VGG or DenseNet) as feature selectors. Further, two separate Python functions, `train.py` and `predict.py` were implemented to train and use the image classifier from the command line or as application.

## Python Libraries
The following Python libraries are key in the project:
* numpy
* torch
* torchvision
* matplotlib
* seaborn

## Key Findings
* neural networks enable to build models for complex tasks, such as image recogniction 
* pre-trained convolutional networks (e.g. VGG or DenseNet) provide excellent feature selectors for image recognition
* the training of the large neural network was sifnificantly speed up by GPU computing
* the implemented neural network reached an accuracy of more than 70% in only 5 epochs

