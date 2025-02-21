"""

    This code prepares the target model. (2/n)

    Firstly, datasets for training and testing the model are constructed. The basis for this is the CIFAR-10 dataset
    with 60 000 images from the ten categories listed as classes below. Each image is a color image 32x32 pixels.
    The dataset is then split into three subsets for the experimental evaluation.

    - 3 000 images from each category are randomly selected for the training dataset, Dtrain.
    - 1 000 images from each category are randomly selected for the test dataset, Dtest.
    - The remaining 2 000 images from each category are stored and will be used later on to simulate an attack
      on the target model.

    Each of the datasets are stored to disk.

    Note that CIFAR-10 is based on the work Learning Multiple Layers of Features from Tiny Images, by Alex Krizhevsky,
    2009, and can be found here: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

    Secondly, a ResNet18-model is loaded from the Torchvision library. For this work, a non-pretrained model is
    used. Documentation for the model is presented here:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18

"""

import os, sys, datetime, pickle, shutil, random

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
    path_substitute_dataset = "surrogate-dataset/"

    # 3 out of 6 are going to training-dataset, 1 out of 6 are going to test-dataset, and 2 out of 6 are going
    # to the surrogate-dataset. Randomizing the the names.

    filelist = []

    for c in classes:
        filelist.clear()
        print(path_original_dataset+c)
        for file in os.listdir(path_original_dataset+c):
            filelist.append(file)

        random.shuffle(filelist)

        train = filelist[0:3000]
        test = filelist[3000:4000]
        substitute = filelist[4000:6000]

        for file in train:
            src = path_original_dataset + c + "/" + file
            dst = path_training_dataset + c + "/" + file
            shutil.copy(src, dst)

        for file in test:
            src = path_original_dataset + c + "/" + file
            dst = path_test_dataset + c + "/" + file
            shutil.copy(src, dst)

        for file in substitute:
            src = path_original_dataset + c + "/" + file
            dst = path_substitute_dataset + c + "/" + file
            shutil.copy(src, dst)


        print(len(train), len(test), len(substitute))

