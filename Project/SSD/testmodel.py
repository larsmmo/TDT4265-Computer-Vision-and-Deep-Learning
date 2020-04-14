import torch
from torch import nn
from torchvision import models
from torchsummary import summary

if __name__ == "__main__":
    model = models.resnext101_32x8d(pretrained = True)

    features = nn.Sequential(*list(model.children())[:-2])

    print(model)