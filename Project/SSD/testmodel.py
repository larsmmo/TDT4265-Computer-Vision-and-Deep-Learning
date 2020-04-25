import torch
from torch import nn
from torchvision import models
from torchsummary import summary

if __name__ == "__main__":
    model = models.resnet34(pretrained = True)

    #features = nn.Sequential(*list(model.children())[:-2])

    for i in range(1, 8):
    	print(i)
    	sk = 0.2 + (0.7/6)*(i-1)
    	print("Width: ")
    	print(sk * 512)
    	print("\n")
    	print("height: ")
    	print(sk * 384)
    	print("\n")
    	
    #print(model)
    #summary(model, (3, 512, 384))
