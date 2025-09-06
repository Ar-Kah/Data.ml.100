# 1. tf_mnist_dump_data.py - Loads MNIST data using TF and dumps as pickle
# 2. tf_mnist_illustrate.py - Loads dump and shows images
import pickle

import matplotlib.pyplot as plt

from random import random

# Read MNIST data from dumped pickle
data_fname = 'mnist_fashion.pkl'

with open(data_fname,'rb') as data_file:
    x_train = pickle.load(data_file)
    y_train = pickle.load(data_file)
    x_test = pickle.load(data_file)
    y_test = pickle.load(data_file)


# Print the size of training and test data
print(f'x_train shape {x_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'x_test shape {x_test.shape}')
print(f'y_test shape {y_test.shape}')

true_class = []
predicted_class = []

for i in range(x_test.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(x_test[i], cmap='gray_r')
        plt.title(f"Image {i} label num {y_test[i]} predicted {0}")
        true_class.append(y_test[i])
        predicted_class.append(0)
        plt.pause(1)
