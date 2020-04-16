import torch
from torch import nn
from torchvision import models
#from torchsummary import summary

class ConvBnRelu(torch.nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=kernel_size, 
                              padding=padding, 
                              stride=stride, 
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AddedLayers(torch.nn.Module):
    def __init__(self, in_ch):
        super(AddedLayers, self).__init__()

        self.convLayer1_1 = ConvBnRelu(in_ch, 256, kernel_size=1, padding=0, stride=1)
        self.convLayer1_2 = ConvBnRelu(256, 512, kernel_size=3, padding=1, stride=1) 
        self.convLayer2_1 = ConvBnRelu(512, 256, kernel_size=1, padding=0, stride=1)
        self.convLayer2_2 = ConvBnRelu(256, 512, kernel_size=3, padding=1, stride=2)
        self.convLayer3_1 = ConvBnRelu(512, 256, kernel_size=1, padding=0, stride=1)
        self.convLayer3_2 = ConvBnRelu(256, 512, kernel_size=3, padding=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self,x):
        out1_1 = self.convLayer1_1(x)
        out1_2 = self.convLayer1_2(out1_1)
        out2_1 = self.convLayer2_1(out1_2)
        out2_2 = self.convLayer2_2(out2_1)
        out3_1 = self.convLayer2_1(out2_2)
        out3_2 = self.convLayer2_2(out3_1)

        out_avg = self.avgpool(out3_2)

        return out2_2, out3_2, out_avg



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

        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.model = models.resnet50(pretrained = True)

        #summary(self.model, (3, 370, 260))

        self.extraLayers = AddedLayers(output_channels[2])
        """
        for param in self.model.parameters(): # Freeze all parameters while training on waymo
            param.requires_grad = False

        for param in self.model.layer4.parameters():    # Unfreeze some of the last convolutional
            param.requires_grad = True                  # layers
        """


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

        out0 = self.model.layer2(x)
        out_features.append(out0)

        out1 = self.model.layer3(out0)
        out_features.append(out1)

        out2 = self.model.layer4(out1)
        out_features.append(out2)

        out3, out4, out5 = self.extraLayers(out2)

        out_features.append(out3)
        out_features.append(out4)
        out_features.append(out5)

        """
        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """
        return tuple(out_features)

