from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *
import torchvision.transforms as transforms


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            #SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            Normalize(cfg.INPUT.PIXEL_MEAN, [0.229*255, 0.224*255, 0.225*255]),
            ToTensor(),
        ]

        if cfg.DATASETS.TRAIN[0].startswith("tdt4265"):
            transform.insert(3, PhotometricDistort())
            
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            #SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            Normalize(cfg.INPUT.PIXEL_MEAN, [0.229*255, 0.224*255, 0.225*255]),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
