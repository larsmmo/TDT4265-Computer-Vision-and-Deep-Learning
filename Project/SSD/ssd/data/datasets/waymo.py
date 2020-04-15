import torch
import pathlib
import numpy as np
import json
from ssd.container import Container
from PIL import Image


class WaymoDataset(torch.utils.data.Dataset):

    class_names = (
        "__background__",
        "vehicle",
        "person",
        "sign",
        "cyclist")
    validation_percentage = 0.15

    def __init__(self, data_dir: str, split: str, transform=None, target_transform=None):
        data_dir = pathlib.Path(data_dir)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_ids = self.read_image_ids(split)
        self.labels = self.read_labels(data_dir.joinpath("labels.json"))
        self.validate_dataset()
        self.image_ids = self.filter_images()
        print(len(self.image_ids))
        self.image_ids = self.split_dataset(split)
        print(f"Dataset loaded. Subset: {split}, number of images: {len(self)}")

    def validate_dataset(self):
        for image_id in self.image_ids:
            assert image_id in self.labels,\
                f"Did not find label for image {image_id} in labels"

    def read_labels(self, label_path):
        assert label_path.is_file(), \
            f"Did not find label file: {label_path.absolute()}"
        with open(label_path, "r") as fp:
            labels = json.load(fp)
        labels_processed = {}
        for label in labels:
            image_id = label["image_id"]
            labels_processed[image_id] = label
        return labels_processed

    def __getitem__(self, index):
        boxes, labels = self.get_annotation(index)
        image = self._read_image(index)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        target = Container(boxes=boxes, labels=labels)
        return image, target, index

    def __len__(self):
        return len(self.image_ids)

    def split_dataset(self, split):
        num_train_images = int(len(self.image_ids) * (1-self.validation_percentage))
        if split == "train":
            image_ids = self.image_ids[:num_train_images]
        else:
            image_ids = self.image_ids[num_train_images:]
        return image_ids

    def read_image_ids(self, split):
        images = self.data_dir.joinpath("images").glob("*.jpg")
        image_ids = [int(x.stem) for x in images]
        image_ids.sort()

        return image_ids

    def _get_annotation(self, image_id):
        label = self.labels[image_id]
        bbox_key = "bounding_boxes"
        if bbox_key not in label:
            bbox_key = "bboxes"
        boxes = np.zeros((len(label[bbox_key]), 4), dtype=np.float32)
        labels = np.zeros((len(label[bbox_key])), dtype=np.int64)
        for idx, bounding_box in enumerate(label[bbox_key]):
            box = [
                bounding_box["xmin"],
                bounding_box["ymin"],
                bounding_box["xmax"],
                bounding_box["ymax"]]
            boxes[idx] = box
            labels[idx] = bounding_box["label_id"]
        # SSD use label 0 as the background. Therefore +1
        labels = labels + 1
        return boxes, labels

    def get_annotation(self, index):
        image_id = self.image_ids[index]
        return self._get_annotation(image_id)

    def _read_image(self, index):
        image_id = self.image_ids[index]
        image_path = self.data_dir.joinpath("images").joinpath(f"{image_id}.jpg")
        image = Image.open(str(image_path)).convert("RGB")
        image = np.array(image)
        return image

    def filter_images(self):
        return self.remove_empty_images()

    def remove_empty_images(self):
        """
            Removes any images without objects for training
        """
        keep_idx = []
        for idx in range(len(self)):
            boxes, labels = self.get_annotation(idx)
            if len(labels) == 0:
                continue
            keep_idx.append(idx)
        return [self.image_ids[idx] for idx in keep_idx]

    def get_img_info(self, index):
        return {"height": 960, "width": 1280}