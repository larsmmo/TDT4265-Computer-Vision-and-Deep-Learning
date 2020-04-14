import torch
from torch import nn
from torchvision import models
from torchsummary import summary

def extraLayer(in_ch, out_ch, num_filters, stride1, stride2, padding1, padding2, kern_size):
    extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=num_filters * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=num_filters * 2), 
            nn.ReLU(inplace = True), 
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters,
                kernel_size=3,
                stride=stride1,
                padding=padding1,
                bias=False,
                groups=32
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kern_size,
                stride=stride2,
                padding=padding2,
                bias=False,
                groups=32
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace = True), 
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=stride1,
                padding=padding1,
                bias=False,
                groups=32
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace = True), 
        )
    return extractor

class ResNextModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()

        self.out_features = []

        def hook_fn(module, input, output):
            self.out_features.append(output.data)
            print("appending hook")

        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.model = models.resnet101(pretrained = True)
        self.model.fc = nn.Sequential()             # Remove fully-connected layer

        summary(self.model, (3, 370, 260))

        self.addedLayer1 = extraLayer(in_ch=output_channels[2],
                                      out_ch=output_channels[3],
                                      num_filters=1024,
                                      stride1=1, 
                                      stride2=2, 
                                      padding1=1, 
                                      padding2=0,
                                      kern_size= 2)
        
        self.addedLayer2 = extraLayer(in_ch=output_channels[3],
                                      out_ch=output_channels[4],
                                      num_filters=512,
                                      stride1=1, 
                                      stride2=2, 
                                      padding1=1, 
                                      padding2=1,
                                      kern_size=3)

        for param in self.model.parameters(): # Freeze all parameters while training on waymo
            param.requires_grad = False
        """
        for param in self.model.layer4.parameters():    # Unfreeze some of the last convolutional
            param.requires_grad = True                  # layers
        for param in self.model.layer3.parameters():    # Unfreeze some of the last convolutional
            param.requires_grad = True                  # layers
        """
        
        #self.model.layer4 = nn.Sequential(self.model.layer4)
        #self.model.layer4.add_module("new1", self.addedLayer1)
        self.model.avgpool = nn.Sequential()
        self.model.avgpool.add_module("extraLayer1", self.addedLayer1)
        self.model.avgpool.add_module("extraLayer2", self.addedLayer2)
        self.model.avgpool.add_module("avgpool2d", nn.AdaptiveAvgPool2d(output_size=(1,1)))
        #self.model.features.children().insert(self.addedLayer2)layer4.add_module("new2", self.addedLayer2)

        print(self.model)

        self.hook1 = self.model.layer2.register_forward_hook(hook_fn)
        self.hook2 = self.model.layer3.register_forward_hook(hook_fn)
        self.hook3 = self.model.layer4.register_forward_hook(hook_fn)

        for name, modul in self.model.avgpool.named_modules():
            if name == "extraLayer1":
                self.hook4 = modul.register_forward_hook(hook_fn)
            if name == "extraLayer2":
                self.hook5 = modul.register_forward_hook(hook_fn)

        #self.hook6 = self.model.avgpool.register_forward_hook(hook_fn)


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        self.out_features = []
        out = self.model(x)
        print("appenging out \n")
        out_features.append(out)
        #out_features = []
        #out_features.append(self.hook1.output)
        #out_features.append(self.hook2.output)
        #out_features.append(self.hook3.output)
        #out_features.append(self.hook4.output)

        #out_features.append(self.model.features[:])
        """
        for idx, feature in enumerate(self.out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """
        return tuple(self.out_features)

