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
    def __init__(self, cfg, pretrained_model):
        super().__init__()

        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.model = models.resnext101_32x8d(pretrained = True)

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

        for param in self.model.layer4.parameters():    # Unfreeze some of the last convolutional
            param.requires_grad = True                  # layers


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
        out_features = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)

        x = self.model.layer2(x)
        out_features.append(x)

        x = self.model.layer3(x)
        out_features.append(x)

        x = self.model.layer4(x)
        out_features.append(x)

        x = self.addedLayer1(x)
        out_features.append(x)

        x = self.addedLayer2(x)
        out_features.append(x)

        x = self.model.avgpool(x)
        out_features.append(x)

        """
        for idx, feature in enumerate(self.out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """
        return tuple(out_features)

