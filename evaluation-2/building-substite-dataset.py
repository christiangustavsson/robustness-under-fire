"""
    A theft of the target model is simulated pulling data from the surrogate dataset.
    This is done once, and stored in a substitute dataset. This will result in a somewhat flawed dataset,
    since the target dataset is not 100% accurate in its classification.

    This is only run once.

"""

from tqdm import tqdm
import numpy as np
import os

import torch
import torchvision
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder

# For handling non-relevant warnings
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# Define data transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":
    model = resnet18(weights = None)
    model.load_state_dict(torch.load('evaluation-2/models/target-model/target-model.pth', map_location=device))
    model.eval().to(device)

    surrogate_dataset = ImageFolder(root='evaluation-2/datasets/surrogate-dataset')

    with torch.no_grad():
        accuracy = 0
        datasamples = len(surrogate_dataset)
        savecat = 'evaluation-2/datasets/substitute-dataset/'

        # Only making predictions and calculating accuracy
        for i in tqdm(range(len(surrogate_dataset))):

            image = surrogate_dataset[i][0]
            classidx = surrogate_dataset[i][1]

            x = transform(image).to(device)
            x.unsqueeze_(0)

            output = model(x)
            _, predicted = torch.max(output.data, 1)

            if classidx == predicted.item():
                accuracy += 1

            savepath = savecat + classes[predicted.item()] + '/subs' + str(i+1) + '.png'
            image.save(savepath)

        print(f"Accuracy of target model predictions to base substitute model on: {100 * accuracy / datasamples} %")

