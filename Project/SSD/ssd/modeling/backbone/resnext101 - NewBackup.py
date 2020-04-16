import torch
from torch import nn
from torchvision import models
#from torchsummary import summary

##############################################
# From torchvision implementation of resnet: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

##############################################

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

        self.extraLayer1 = _make_extra_layer(Bottleneck, output_channels[2], 1)
        self.extraLayer2 = _make_extra_layer(Bottleneck, output_channels[3], 1)
        self.extraLayer3 = _make_extra_layer(Bottleneck, output_channels[4], 1)

        #self.extraLayers = AddedLayers(output_channels[2])

        for m in self.extraLayers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        """
        for param in self.model.parameters(): # Freeze all parameters while training on waymo
            param.requires_grad = False

        for param in self.model.layer4.parameters():    # Unfreeze some of the last convolutional
            param.requires_grad = True                  # layers
        """

    ## The following function is from the pytroch resnet implementation:
    def _make_extra_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

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

