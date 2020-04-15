import torch
import pathlib
import numpy as np
from .mnist_object_detection.mnist_object_dataset import load_dataset
from ssd.container import Container


class MNISTDetection(torch.utils.data.Dataset):

    class_names = ["__background__"] + [str(x) for x in range(10)]

    def __init__(self, data_dir: str, is_train: bool, transform=None, target_transform=None):
        data_dir = pathlib.Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.images, labels, bboxes_XYXY = load_dataset(data_dir, is_train)
        self.bboxes_XYXY = bboxes_XYXY
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        boxes, labels = self.get_annotation(index)
        image = image[:, :, None].repeat(3, -1)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        target = Container(boxes=boxes, labels=labels)
        
        return image, target, index

    def __len__(self):
        return len(self.images)
    
    def get_img_info(self, index):
        #width, height = self.images[index].size
        height, width = self.images.shape[1:3]
        return {"height": height, "width": width}

    def get_annotation(self, index):
        boxes = self.bboxes_XYXY[index].copy().astype(np.float32)
        # SSD use label 0 as the background. Therefore +1
        labels = self.labels[index].copy().astype(np.int64) + 1
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))
