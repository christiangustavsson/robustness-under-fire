"""

    This code prepares the target model. (1/n)

    From the downloaded cifar-10 files, the training and test files are gathered in their categories. This is done
    to randomize them for this work.

    Note that CIFAR-10 is based on the work Learning Multiple Layers of Features from Tiny Images, by Alex Krizhevsky,
    2009, and can be found here: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
"""

import os, sys, datetime, pickle, shutil
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# Classes
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = len(classes)

if __name__ == "__main__":

    # Reading and splitting datasets
    path_original_dataset = "original-dataset/" # Relative paths
    path_training_dataset = "training-dataset/"
    path_test_dataset = "test-dataset/"

    # 3 out of 6 are going to training-dataset, 1 out of 6 are going to test-dataset, and 2 out of 6 are going
    # to the surrogate-dataset. Randomizing the the names.

    ordering = np.arange(0, 60001, 1)
#    np.random.shuffle(ordering)

    i = 0

    for c in classes:
        for file in os.listdir(path_original_dataset + "train/" + c):
            number = ordering[i]
            src = path_original_dataset + "train/" + c + "/" + file
            dst = path_original_dataset + c + "/image" + str(number) + ".png"
            print(src + " -> " + dst)
            shutil.copy(src, dst)
            i += 1

        for file in os.listdir(path_original_dataset + "test/" + c):
            number = ordering[i]
            src = path_original_dataset + "test/" + c + "/" + file
            dst = path_original_dataset + c + "/image" + str(number) + ".png"
            print(src + " -> " + dst)
            shutil.copy(src, dst)
            i += 1


