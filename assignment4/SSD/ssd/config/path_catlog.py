import os


class DatasetCatalog:
    DATA_DIR = ''
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOCdevkit/VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOCdevkit/VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOCdevkit/VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOCdevkit/VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
        'mnist_detection_train': {
            'data_dir': 'mnist_detection/train',
            'split': 'train'
        },
        'mnist_detection_val': {
            'data_dir': 'mnist_detection/test',
            'split': 'val'
        }
    }

    @staticmethod
    def get(base_path, name):
        root = os.path.join(base_path, DatasetCatalog.DATA_DIR)
        if "voc" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root, attrs["data_dir"]),
                ann_file=os.path.join(root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)
        elif "mnist_detection" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root, attrs["data_dir"]),
                is_train= attrs["split"] == "train"
            )
            return dict(factory="MNISTDetection", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
