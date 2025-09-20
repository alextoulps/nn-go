#!/bin/bash

mkdir -p datasets

wget -q --show-progress https://python-course.eu/data/mnist/mnist_train.csv -O datasets/mnist_train.csv
wget -q --show-progress https://python-course.eu/data/mnist/mnist_test.csv -O datasets/mnist_test.csv

head -n 501 datasets/mnist_train.csv > datasets/mnist_train_500.csv
head -n 1001 datasets/mnist_train.csv > datasets/mnist_train_1000.csv

head -n 11 datasets/mnist_test.csv > datasets/mnist_test_10.csv
head -n 21 datasets/mnist_test.csv > datasets/mnist_test_20.csv
head -n 51 datasets/mnist_test.csv > datasets/mnist_test_50.csv

ls -lh datasets/*.csv