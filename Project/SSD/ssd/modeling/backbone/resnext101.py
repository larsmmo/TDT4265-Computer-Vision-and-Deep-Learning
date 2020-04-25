import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
#from torchsummary import summary

##############################################
# From torchvision implementation of resnet: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, padding = 1, norm_layer=None):
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
        self.conv2 = conv3x3(planes, planes, stride = 1, groups = 1, dilation=1, padding=padding)
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
                 base_width=64, dilation=1, padding=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation=dilation, padding=padding)
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
# Light scratch network inspired by: https://github.com/vaesl/LRF-Net/blob/master/models/LRF_COCO_300.py
#http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Rich_Features_at_High-Speed_for_Single-Shot_Object_Detection_ICCV_2019_paper.pdf

class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu = True):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False) 
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class LargeDownsampler(torch.nn.Module):
    def __init__(self,):
        self.maxPool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.maxPool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.maxPool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)

    def forward(self, x):
        x = self.pool1(x)
        x = self.pool2(x)
        x_out = self.pool3(x)
        return x_out

class LightScratchNetwork(torch.nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1):
        super(LightScratchNetwork, self).__init__()
        inter_planes = out_planes // 4
        self.downsampler = LargeDownsampler()
        self.commonConvs = nn.Sequential(
                ConvBlock(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1),
                ConvBlock(inter_planes, inter_planes, kernel_size=1, stride=1),
                ConvBlock(inter_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1)
                )
        self.conv1_out = ConvBlock(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)

        self.conv2 = ConvBlock(inter_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1)
        self.conv2_out = ConvBlock(inter_planes, out_planes * 2, kernel_size=1, stride=1, relu=False)

        self.conv3 = ConvBlock(inter_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1)
        self.conv3_out = ConvBlock(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)

        #self.conv4 = ConvBlock(inter_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1)
        #self.conv4_out = ConvBlock(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)

    def forward(self,x):
        x = self.downsampler(x)     # Downsample image
        x = self.commonConvs(x) 
        out1 = self.conv1_out(x)
        x = self.conv2(x)
        out2 = self.conv2_out(x)
        x = self.conv3(x)
        out3 = self.conv3_out(x) 

        return out1, out2, out3

class TopDownModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TopDownModule, self).__init__()
        self.lateral_layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.smooth_layer = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, bigFeature, smallFeature):
        _,_,H,W = bigFeature.size()

        x = F.interpolate(smallFeature, size=(H,W), mode='bilinear', align_corners=True) + self.lateral_layer(bigFeature)
        x = self.smooth_layer(x)

        return x


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
        super(ResNextModel, self).__init__()

        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        # Backbone model
        self.model = models.resnet34(pretrained = True)

        # Top-down modules for feature pyramid network
        #self.TD1 = TopDownModule(output_channels[4], output_channels[5])
        #self.TD2 = TopDownModule(output_channels[3], output_channels[4])
        #self.TD3 = TopDownModule(output_channels[2], output_channels[3])
        #self.TD4 = TopDownModule(output_channels[1], output_channels[2])
        #self.TD5 = TopDownModule(256, output_channels[1])

        """
        # Light-weight scratch network
        self.LSN = LightScratchNetwork(image_channels, 512)

        # Convs with downsampling for bottom-up scheme
        self.ds_conv1 = ConvBlock(output_channels[0], out_planes, kernel_size=(3, 3), stride=stride, padding=padding, relu=False)
        self.ds_conv2 = ConvBlock(output_channels[1], out_planes, kernel_size=(3, 3), stride=stride, padding=padding, relu=False)
        self.ds_conv3 = ConvBlock(output_channels[2], out_planes, kernel_size=(3, 3), stride=stride, padding=padding, relu=False)
        """

        self.inplanes = output_channels[2]
        self.groups = 1
        self.base_width = 64
        self.dilation = 1

        #summary(self.model, (3, 370, 260))
        # Adding extra layers for smaller feature maps: Residual blocks with downsampling
        
        self.extraLayers = nn.Sequential(
            self._make_extra_layer(BasicBlock, 512, 512, 1, stride = 2),
            self._make_extra_layer(BasicBlock, 512, 512, 1, stride = 2),
            self._make_extra_layer(BasicBlock, 512, 256, 1, stride = 2, padding=1),
            self._make_extra_layer(BasicBlock, 256, 256, 1, stride = 2, padding=1),
            self._make_extra_layer(BasicBlock, 256, 256, 1, stride = 2, padding=1)
            #nn.AdaptiveAvgPool2d((1,1))
            #self._make_extra_layer(BasicBlock, 512, 1, stride = 1, padding=0)
        )
        """
        nn.Sequential(
            conv3x3(512, 256, stride=1, groups=1,dilation=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3,4), stride=1,
                 padding=0, groups=1, bias=False, dilation=1)
            #conv3x3(256, 512,stride=1,groups=1,dilation=1,padding=0)
            )
        """

        #self.extraLayers = AddedLayers(output_channels[2])
        #print(self.extraLayers[2])

        # Initialize weights in extra layers with kaiming initialization
        for layer in self.extraLayers:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        """
        for param in self.model.parameters(): # Freeze all parameters while training on waymo
            param.requires_grad = False

        for param in self.model.layer3.parameters():    # Unfreeze some of the last convolutional
            param.requires_grad = True                  # layers

        for param in self.model.layer4.parameters():    # Unfreeze some of the last convolutional
            param.requires_grad = True                  # layers
        """


    ## The following function is from the pytroch resnet implementation (modified):
    def _make_extra_layer(self, block, in_planes, out_planes, blocks, stride=1, dilate=False, padding=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1: #or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(in_planes, out_planes * block.expansion, stride),
                norm_layer(out_planes * block.expansion),
            )

        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample, self.groups,
                            self.base_width, dilation=previous_dilation, padding = padding, norm_layer=norm_layer))
        #self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_planes, out_planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, padding = padding,
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

        #lsn_out1, lsn_out2, lsn_out3 = self.LSN(x)      # Pass image through light-weight scratch network

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        #x = self.model.maxpool(x)
        x = self.model.layer1(x)

        x = self.model.layer2(x)
        #out_features.append(out0)

        feature1 = self.model.layer3(x)
        feature2 = self.model.layer4(feature1)
        feature3 = self.extraLayers[0](feature2)
        feature4 = self.extraLayers[1](feature3)
        feature5 = self.extraLayers[2](feature4)
        feature6 = self.extraLayers[3](feature5)

        feature7 = self.extraLayers[4](feature6) 

        #p3 = self.TD3(feature3, feature4)
        #p2 = self.TD4(feature2, p3)
        #p1 = self.TD5(feature1, p2)
        
        """
        out_features.append(p1)
        out_features.append(p2)
        out_features.append(p3)
        out_features.append(p4)
        out_features.append(p5)
        out_features.append(feature6)
        
        
        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx][1], self.output_feature_size[idx][0])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        print("Passed tests")
        """
        return (feature1, feature2, feature3, feature4, feature5, feature6, feature7)
        #return (p1, p2, p3, feature4, feature5, feature6)

