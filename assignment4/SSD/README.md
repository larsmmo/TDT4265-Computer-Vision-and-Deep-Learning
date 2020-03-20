# SSD300 In Pytorch 1.3
This implementation is a slimmed down version of: https://github.com/lufficc/SSD
Thanks to the original authors for creating the amazing repository and setting a MIT license on it.

### Features
This code base will be the start for your assignment 4 and the final project.
You can customize a ton of stuff with it, for example:

1. There exists a bunch of already implemented data augmentation techniques in [transforms.py](ssd/data/transforms/transform.py)
2. There is an extensive config file that you can set. For example, in the .yml files in configs/, we can override the defaults. To check out the default config, see: [defaults.py](ssd/config/defaults.py)
3. Three datasets are currently supported: MNIST Object detection, PASCAL VOC, and COCO.
4. Tensorboard support. Everything is logged to tensorboard, and you can check out the logs in either the [custom notebook](plot_scalars.ipynb), or launching a tensorboard with the command: `tensorboard --logdir outputs`


### Setup
On local computer, using the environment you previously had.
Install Pytorch >= 1.3 and torchvision>= 0.4.0 (You can check your versions with `pip list`).

Then, install the requirements from this directory:
```
pip install -r requirements.txt
```
If you are on the Cybele/tulipan computers, use:
```
pip install -r --user requirements.txt
```

The environment is already fixed on the tdt4265.idi.ntnu.no server!

#### Tensorboard:
To start logging on tensorboard, do:
```
tensorboard --logidr outputs
```
BTW, you can also check this out in the jupyter lab notebook on the server, by clicking this:

![](https://raw.githubusercontent.com/chaoleili/jupyterlab_tensorboard/master/image/launcher.png)


## Train
Train for:
1. MNIST:
```bash
python train.py  configs/mnist.yaml
```
2. MNIST on the tdt4265.idi.ntnu.no server (This only changes the "datasets" variable in the config file):
```bash
python train.py  configs/mnist_tdt4265_server.yaml
```
3. VGG SSD300 on the PACAL VOC dataset
```bash
python train.py  configs/vgg_ssd300_voc0712.yaml
```

## Evaluate
Run test.py to evaluate the whole validation dataset: 
```bash
python test.py configs/mnist.yaml
```
Remember to give the correct config path

## Demo
For Pascal VOC
```bash
python demo.py configs/vgg_ssd300_voc0712.yaml --images_dir demo/voc --dataset_type voc
```

For MNIST Object detection
```bash
python demo.py  configs/mnist.yaml --images_dir demo/voc --dataset_type mnist
```

You will see a similar output:
```text
(0001/0005) 004101.jpg: objects 01 | load 010ms | inference 033ms | FPS 31
(0002/0005) 003123.jpg: objects 05 | load 009ms | inference 019ms | FPS 53
(0003/0005) 000342.jpg: objects 02 | load 009ms | inference 019ms | FPS 51
(0004/0005) 008591.jpg: objects 02 | load 008ms | inference 020ms | FPS 50
(0005/0005) 000542.jpg: objects 01 | load 011ms | inference 019ms | FPS 53
```

### Setting Up Datasets (Local)
Check the links here:
- [PASCAL VOC](pascal.md)
- [COCO Dataset](coco.md)
- MNIST dataset will be downloaded automatic.

