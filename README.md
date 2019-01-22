# An implementation of ResNet using Numpy
## Introduction
This is my finished project for my course *Introduction to Machine Learning* taught by *Prof. Wang Jie* at USTC. It is an simple implementation of ResNet34 using Python Numpy. It supports the training and inference process on part of tiny ImageNet dataset (in the folder *dataset*). If you need more information about dataset or the course please refer to http://staff.ustc.edu.cn/~jwangx/classes/210709/project.html
## Use guide
Run the script *train.py* to train on offered dataset. You can set hyper parameters in the script *train.py*. Also, if you would like to train on your own dataset, you must prepare a *.txt* file in the format of *train.txt* or *test.txt* first to enable the program to be access to your data. Notice this project **does not support GPU computing** so the training process can be very long on big dataset.

We have offered trained model in the folder *model*. You can run the script *demo_test.py* to test it on our test set.   