# Setting Up Datasets
Note that all datasets are downloaded on the tdt4265.idi.ntnu.no server and the tulipan/cybele computers (saved under /work/datasets).


Index
- [TDT4265 Dataset](#tdt4265-dataset)
- [Waymo Dataset](#waymo)
- Datasets **not** used in the project:
    - [PASCAL VOC](#pascal-voc)
    - [COCO Dataset](#coco)
    - MNIST dataset will be downloaded automatic. For information, see: [hukkelas/MNIST-ObjectDetection](github.com/hukkelas/MNIST-ObjectDetection)

---

## Waymo Dataset

**It is not allowed to use this dataset outside of this class. It is not allowed to redistribute the dataset and you are only allowed to use it for academic and educational purposes.**

To take a peek at the dataset, take a look at [visualize_dataset.ipynb](../visualize_dataset.ipynb) or [visualize_dataset.py](../visualize_dataset.py).

The [waymo dataset](https://waymo.com/open/about/) is a high resolution dataset for autonomous vehicle research.
The dataset consists of a large range of sensor data, such as LIDAR, cameras, and vehicle information. However, for this class we will only use the 2D bounding box data and a subset of the entire dataset.

#### Getting started
- Download:
    - Run the file `python3 setup_waymo.py`
    - For the TDT4265 server and cybele computers, the dataset is locally available and the script will automatically setup the dataset.
    - For local computers, the script will download the dataset. Then it will set up everything.
- Change the following variables in the config file (This is already done in [train_waymo.yaml](../configs/train_waymo.yaml)):
```
DATASETS:
    TRAIN: ("waymo_train",)
    TEST: ("waymo_val", )
```

#### Dataset Information
- We've created dataset splits; train and validation. The validation is 20% of the entire dataset. This can be changed in [waymo.py](../ssd/data/datasets/waymo.py)
- Consists of three classes: vehicle, person and cyclist. 
- Each image has a resolution of 1280 x 960 
- Something to note is that we've set number of classes to 4+1 (even though it should be 3+1) in the waymo config file. This is such that we can keep the same output layers for the Waymo dataset as the TDT4265 dataset. This will cause the **"sign" mAP to show nan** in the evaluation process, but that should be fine.


---

## TDT4265 Dataset

To take a peek at the dataset, take a look at [visualize_dataset.ipynb](../visualize_dataset.ipynb) or [visualize_dataset.py](../visualize_dataset.py).

#### Getting started
- Download:
    - Everything will be done automatically with a script:  `python3 update_tdt4265_dataset.py`.
    - **REQUIRED TO READ** For the TDT4265 server and cybele computers, the dataset is locally available and the script will automatically setup the dataset.
    - This will take some time on computers that are not provided by us, since you need to download the dataset (which is about 5GB).
    - We recommend you to **run this file once in a while**, since labels will be automatically updated as students annotate more data.
- Change the following variables in the config file (This is already done in [train_tdt4265.yaml](../configs/train_tdt4265.yaml)):
```
DATASETS:
    TRAIN: ("tdt4265_train",)
    TEST: ("tdt4265_val", )
```


#### Dataset Information
- Every image is from the streets of trondheim.
- The dataset consists of three subsets; *train*, *validation*, and *test*. For training and validation you have the labels, however, the test set labels are unkown, and used to evaluate the project.
You can set the size of the validation set in [the dataset](../ssd/data/datasets/tdt4265.py).
- The data is collected from 35 FPS video, where we extracted every frame for each video. However, to reduce the amount of data required to download, we've given you a subset containing 7 frames per second from the original video.
- Each image has a resolution of 1280 x 960
- The dataset contains both automatic and non-automatic labels (indicated in the json label file). The automatic annotation model is Faster R-CNN model trained on the COCO dataset.

Label format:
```json
[{
    "image_id": 0, 
    "annotation_completed": True, 
    "video_id": 0, 
    "is_test": True, 
    "bounding_boxes": [
        {
            "ymin": 489,
            "ymax": 500,
            "xmin": 50,
            "xmax": 90,
            "label_id": 2, 
            "label": "person", 
        }
    ]
}]
```

---

## Pascal VOC
For Pascal VOC dataset, make the folder structure like this:
```
VOCdevkit
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ ...
```

You can download this if you are on the school network directly as a .zip file (Note that this dataset should be only used for educational/academic purposes).

With scp: 
```
scp [ntnu-username]@oppdal.idi.ntnu.no:/work/datasets/VOC.zip .

unzip VOC.zip
```


Or you can download it from the PASCAL VOC website:
http://host.robots.ox.ac.uk/pascal/VOC/

Note that we are using the VOC2007 TRAIN/VAL + VOC2012 TRAIN/VAL as the train set.

We use VOC2007 Test as the validation set.



## COCO
This is not used for assignment 4 nor the project.


For COCO dataset, make the folder structure like this:
```
COCO_ROOT
|__ annotations
    |_ instances_valminusminival2014.json
    |_ instances_minival2014.json
    |_ instances_train2014.json
    |_ instances_val2014.json
    |_ ...
|__ train2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ val2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ ...
```
