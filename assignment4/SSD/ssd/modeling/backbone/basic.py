import torch
from torch import nn

def reluConvLayers(in_ch, out_ch, num_filters, stride1, stride2, padding1, padding2):
    extractor = nn.Sequential(
            nn.BatchNorm2d(num_features=in_ch),
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=num_filters,
                kernel_size=3,
                stride=stride1,
                padding=padding1
            ),
            nn.BatchNorm2d(num_features=num_filters), 
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters * 2,
                kernel_size=3,
                stride=stride1,
                padding=padding1
            ),
            nn.BatchNorm2d(num_features=num_filters * 2),
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=out_ch,
                kernel_size=3,
                stride=stride2,
                padding=padding2
            ),
        )
    return extractor

class BasicModel(torch.nn.Module):
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

        num_filters = 32

        self.extraOutput = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),        

            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters * 2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=num_filters * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),               

            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=num_filters * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
        )

        self.output0 = nn.Sequential(
            nn.BatchNorm2d(num_features=num_filters * 2),
            nn.ReLU(), 

            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=num_filters*2),
            nn.ReLU(), 

            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=num_filters*4),
            nn.ReLU(), 

            nn.Conv2d(
                in_channels=num_filters * 4,
                out_channels=output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )

        self.output1 = reluConvLayers(in_ch=output_channels[1],
                                      out_ch=output_channels[2],
                                      num_filters=128,
                                      stride1=1, 
                                      stride2=2, 
                                      padding1=1, 
                                      padding2=1)

        self.output2 = reluConvLayers(in_ch=output_channels[2],
                                      out_ch=output_channels[3],
                                      num_filters=256,
                                      stride1=1, 
                                      stride2=2, 
                                      padding1=1, 
                                      padding2=1)

        self.output3 = reluConvLayers(in_ch=output_channels[3],
                                      out_ch=output_channels[4],
                                      num_filters=128,
                                      stride1=1, 
                                      stride2=2, 
                                      padding1=1, 
                                      padding2=1)

        self.output4 = reluConvLayers(in_ch=output_channels[4],
                                      out_ch=output_channels[5],
                                      num_filters=128,
                                      stride1=1, 
                                      stride2=2, 
                                      padding1=1, 
                                      padding2=1)

        self.output5 = reluConvLayers(in_ch=output_channels[5],
                                      out_ch=output_channels[6],
                                      num_filters=128,
                                      stride1=1, 
                                      stride2=1, 
                                      padding1=1, 
                                      padding2=0)
    
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
        out_features.append(self.extraOutput(x))
        out_features.append(self.output0(out_features[0]))
        out_features.append(self.output1(out_features[1])) 
        out_features.append(self.output2(out_features[2])) 
        out_features.append(self.output3(out_features[3])) 
        out_features.append(self.output4(out_features[4])) 
        out_features.append(self.output5(out_features[5]))

        """
        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """

        return tuple(out_features)

